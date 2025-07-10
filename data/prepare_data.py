"""
Pipeline for downloading and processing PubChem compound data.

This module provides functionality to download SDF files from PubChem,
extract SMILES strings, and perform random sampling for dataset preparation.
"""

import os
import sys
import gzip
import random
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Iterator, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from urllib.request import urlopen, Request
from urllib.error import URLError
import time
import hashlib
import re
import glob
import csv
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Add molecule_utils to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "molecule_utils"))

try:
    from convert import (
        convert_smiles_to_selfies,
        MolecularConversionError,
        InvalidSMILESError
    )
    SELFIES_AVAILABLE = True
except ImportError as e:
    # Use a logger that is configured to be sure the message is seen
    logging.basicConfig(level=logging.INFO)
    logging.warning(f"SELFIES conversion will not be available: {e}")
    SELFIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _worker_convert_smiles(smiles: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Worker function to convert a single SMILES to SELFIES for multiprocessing.

    Returns a tuple of (original_smiles, processed_smiles, selfies_result, error_message).
    This function is defined at the top level to be pickleable.
    """
    if not SELFIES_AVAILABLE:
        return smiles, None, None, "SELFIES library not available in worker process."

    smiles = smiles.strip()
    if not smiles:
        return smiles, None, None, "Empty SMILES string"
        
    try:
        # Use enhanced conversion with preprocessing to avoid hydrogen warnings
        selfies, validation_info = convert_smiles_to_selfies(
            smiles, 
            validate_input=True,
            preprocess=True,  # This will clean up problematic structures
            return_validation_info=True
        )
        processed_smiles = validation_info.get('preprocessed_smiles', smiles)
        return smiles, processed_smiles, selfies, None
    except (InvalidSMILESError, MolecularConversionError) as e:
        # Extract the specific reason if available
        error_msg = str(e)
        if "SMILES failed preprocessing" in error_msg:
            return smiles, None, None, "Failed quality checks"
        return smiles, None, None, error_msg
    except Exception as e:
        return smiles, None, None, f"Unexpected error: {str(e)}"


class PubChemDownloadError(Exception):
    """Custom exception for PubChem download errors."""
    pass


class SDFParsingError(Exception):
    """Custom exception for SDF parsing errors."""
    pass


class PubChemSMILESPipeline:
    """
    Robust pipeline for downloading and processing PubChem compound data.
    
    This class handles downloading SDF files from PubChem, extracting SMILES,
    and performing random sampling for large-scale dataset preparation.
    """
    
    BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"
    
    def __init__(
        self,
        data_dir: str = "data",
        cache_subdir: str = "pubchem",
        output_subdir: str = "output",
        aria2c_connections: int = 5,
        max_retries: int = 5,
        chunk_size: int = 8192,
        debug: bool = False,
        download_timeout: int = 1800
    ):
        """
        Initialize the PubChem SMILES pipeline.
        
        Args:
            data_dir: Main data directory for all data storage.
            cache_subdir: Subdirectory for cached downloads.
            output_subdir: Subdirectory for processed output files.
            aria2c_connections: Number of parallel connections for aria2c.
            max_retries: Maximum number of retry attempts for failed operations.
            chunk_size: Chunk size for file I/O operations.
            debug: Enable debug mode for more verbose logging.
            download_timeout: Download timeout per file in seconds.
        """
        self.data_dir = Path(data_dir).resolve()
        self.cache_dir = self.data_dir / cache_subdir
        self.output_dir = self.data_dir / output_subdir
        self.aria2c_connections = aria2c_connections
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.debug = debug
        self.download_timeout = download_timeout
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup dependency checks
        self._check_aria2c()
        if not PYARROW_AVAILABLE:
            logger.warning("pyarrow is not installed. Parquet processing will be unavailable.")
        
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
        logger.info(f"SELFIES conversion: {'Available' if SELFIES_AVAILABLE else 'Not available'}")

    def _update_progress(
        self,
        collected: int,
        target: int,
        start_time: float,
        prefix: str = 'Progress'
    ) -> None:
        """
        Display a dynamic progress bar in the console.
        Writes to stderr to avoid interfering with stdout redirection.
        """
        elapsed_time = time.time() - start_time
        rate = collected / elapsed_time if elapsed_time > 0 else 0

        # Handle cases with unknown or zero target
        if target <= 0:
            progress_info = f"\r{prefix}: Processed {collected:,} items at {rate:,.0f}/s...   "
            sys.stderr.write(progress_info)
            sys.stderr.flush()
            return
        
        percentage = min(100, (collected / target) * 100)
        
        eta_seconds = (target - collected) / rate if rate > 0 and collected > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds > 0 and eta_seconds < 86400 * 2 else 'N/A'
        
        bar_length = 40
        filled_length = int(bar_length * collected // target)
        bar = '█' * filled_length + '─' * (bar_length - filled_length)
        
        progress_info = (
            f"\r{prefix}: |{bar}| {percentage:.1f}% ({collected:,}/{target:,}) "
            f"Rate: {rate:,.0f}/s, ETA: {eta_str}   "
        )
        sys.stderr.write(progress_info)
        sys.stderr.flush()
    
    def _check_aria2c(self) -> None:
        """Check if aria2c is available and accessible."""
        try:
            result = subprocess.run(
                ["aria2c", "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            logger.info("aria2c dependency check passed.")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise PubChemDownloadError(
                "aria2c is not installed or not in PATH. "
                "Please install it (e.g., 'sudo apt-get install aria2' or 'brew install aria2')."
            )
    
    def get_available_files(self) -> List[Tuple[str, str]]:
        """
        Get list of available SDF files from PubChem FTP.
        
        Returns:
            List of tuples (filename, md5_filename) for available SDF files.
            
        Raises:
            PubChemDownloadError: If unable to fetch file list.
        """
        logger.info("Fetching available file list from PubChem FTP...")
        
        try:
            request = Request(self.BASE_URL, headers={'User-Agent': 'PubChem-Pipeline/1.1'})
            with urlopen(request, timeout=60) as response:
                content = response.read().decode('utf-8')
            
            sdf_files = set(re.findall(r'href="([^"]*\.sdf\.gz)"', content))
            md5_files = set(re.findall(r'href="([^"]*\.sdf\.gz\.md5)"', content))
            
            file_pairs = [
                (sdf_file, md5_file)
                for sdf_file in sdf_files
                if (md5_file := f"{sdf_file}.md5") in md5_files
            ]
            
            if not file_pairs:
                raise PubChemDownloadError("Could not find any SDF files on the PubChem FTP server.")

            logger.info(f"Found {len(file_pairs)} SDF files available for download.")
            return file_pairs
            
        except URLError as e:
            raise PubChemDownloadError(f"Failed to fetch file list from PubChem: {e}")
        except Exception as e:
            raise PubChemDownloadError(f"An unexpected error occurred while fetching file list: {e}")
    
    def download_file_with_aria2c(self, filename: str, md5_filename: str) -> Optional[Path]:
        """
        Download a single SDF file using aria2c with checksum verification.
        
        Args:
            filename: Name of the SDF file to download.
            md5_filename: Name of the corresponding MD5 file.
            
        Returns:
            Path to the downloaded file if successful, otherwise None.
        """
        sdf_url = f"{self.BASE_URL}{filename}"
        md5_url = f"{self.BASE_URL}{md5_filename}"
        
        sdf_path = self.cache_dir / filename
        md5_path = self.cache_dir / md5_filename
        
        if sdf_path.exists() and self._verify_file_integrity(sdf_path, md5_path):
            logger.info(f"File '{filename}' already exists and is valid, skipping download.")
            return sdf_path
        
        try:
            # Download MD5 file
            logger.info(f"Downloading MD5 checksum for {filename}...")
            md5_cmd = [
                "aria2c", "-x1", "-s1", "--dir", str(self.cache_dir), "--out", md5_filename,
                "--retry-wait=3", f"--max-tries={self.max_retries}", "--timeout=60", md5_url
            ]
            subprocess.run(md5_cmd, check=True, capture_output=True)

            # Download SDF file
            file_size_mb = (self._get_remote_file_size(sdf_url) or 0) / (1024 * 1024)
            logger.info(f"Downloading {filename} ({file_size_mb:.1f} MB)...")
            sdf_cmd = [
                "aria2c", f"-x{self.aria2c_connections}", f"-s{self.aria2c_connections}",
                "--min-split-size=10M", "--dir", str(self.cache_dir), "--out", filename,
                "--retry-wait=3", f"--max-tries={self.max_retries}", f"--timeout={self.download_timeout}",
                sdf_url
            ]
            if self.debug:
                sdf_cmd.append("--summary-interval=1")
            
            subprocess.run(sdf_cmd, check=True, capture_output=not self.debug)

            # Verify integrity
            logger.info(f"Verifying integrity of {filename}...")
            if self._verify_file_integrity(sdf_path, md5_path):
                logger.info(f"Successfully downloaded and verified {filename}.")
                return sdf_path
            else:
                logger.error(f"Integrity check failed for {filename}. Deleting corrupted file.")
                sdf_path.unlink(missing_ok=True)
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download '{filename}'. Subprocess error: {e.stderr.decode()}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading {filename}: {e}")
            return None
    
    def _get_remote_file_size(self, url: str) -> Optional[int]:
        """Get the size of a remote file without downloading it."""
        try:
            request = Request(url, method='HEAD', headers={'User-Agent': 'PubChem-Pipeline/1.1'})
            with urlopen(request, timeout=10) as response:
                return int(response.headers.get('Content-Length', 0))
        except Exception:
            return None
    
    def _verify_file_integrity(self, file_path: Path, md5_path: Path) -> bool:
        """Verify file integrity using MD5 checksum."""
        if not file_path.exists() or not md5_path.exists():
            return False
        
        try:
            with open(md5_path, 'r') as f:
                expected_md5 = f.read().strip().split()[0]
            
            actual_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(self.chunk_size), b''):
                    actual_md5.update(chunk)
            
            return actual_md5.hexdigest() == expected_md5
        except Exception as e:
            logger.error(f"Error during integrity verification for {file_path.name}: {e}")
            return False
    
    def extract_smiles_from_sdf(self, sdf_path: Path) -> Iterator[str]:
        """
        Extract SMILES strings from a compressed SDF file.
        
        Args:
            sdf_path: Path to the compressed SDF file.
            
        Yields:
            SMILES strings found in the SDF file.
        """
        try:
            with gzip.open(sdf_path, 'rt', encoding='utf-8', errors='ignore') as f:
                current_compound = []
                in_compound = False
                for line in f:
                    if in_compound:
                        current_compound.append(line)
                        if line.strip() == '$$$$':
                            smiles = self._extract_smiles_from_compound(current_compound)
                            if smiles:
                                yield smiles
                            current_compound = []
                            in_compound = False
                    elif line.strip() == '$$$$':
                        # This handles cases of multiple $$$$ delimiters
                        in_compound = True
                        current_compound = []

        except Exception as e:
            raise SDFParsingError(f"Error parsing SDF file {sdf_path}: {e}")
    
    def _extract_smiles_from_compound(self, compound_lines: List[str]) -> Optional[str]:
        """Extract SMILES from a single compound's SDF record."""
        try:
            for i, line in enumerate(compound_lines):
                if 'PUBCHEM_OPENEYE_CAN_SMILES' in line or 'PUBCHEM_CANONICAL_SMILES' in line:
                    if i + 1 < len(compound_lines):
                        smiles = compound_lines[i + 1].strip()
                        # A simple check for a plausible SMILES string
                        if smiles and not smiles.isspace():
                            return smiles
            return None
        except Exception:
            return None
    
    def convert_smiles_to_selfies_file(
        self,
        input_filename: str,
        output_filename: str,
        num_workers: int,
        chunksize: int = 10000
    ) -> Tuple[int, int]:
        """
        Convert a file of SMILES to SELFIES format using multiple processes.
        
        Args:
            input_filename: Name of the input SMILES file in the output directory.
            output_filename: Name of the output SELFIES file.
            num_workers: Number of worker processes to use.
            chunksize: Number of SMILES to process in each batch.
            
        Returns:
            Tuple of (successful_conversions, failed_conversions).
        """
        if not SELFIES_AVAILABLE:
            raise ValueError("SELFIES conversion is not available. Please install required packages.")
        
        input_filepath = self.output_dir / input_filename
        output_filepath = self.output_dir / output_filename
        failures_filepath = self.output_dir / f"{Path(input_filename).stem}_failures.txt"
        stats_filepath = self.output_dir / f"{Path(input_filename).stem}_conversion_stats.json"
        
        if not input_filepath.exists():
            raise FileNotFoundError(f"SMILES file not found: {input_filepath}")
        
        logger.info(f"Converting SMILES to SELFIES with {num_workers} workers...")
        logger.info(f"Input: {input_filename}, Output: {output_filename}")
        logger.info("Preprocessing is enabled to filter and clean problematic structures.")
        
        successful_conversions = 0
        failed_conversions = 0
        failure_reasons = {}
        processed_smiles_set = set()
        
        # Count total lines for accurate progress reporting
        logger.info("Counting molecules in input file...")
        with open(input_filepath, 'r') as f:
            total_to_process = sum(1 for line in f if line.strip())
        logger.info(f"Found {total_to_process:,} molecules to process.")

        with open(input_filepath, 'r') as input_file, \
             open(output_filepath, 'w', newline='') as output_csv_file, \
             open(failures_filepath, 'w') as failures_file, \
             ProcessPoolExecutor(max_workers=num_workers) as executor:

            writer = csv.writer(output_csv_file)
            writer.writerow(["SELFIES", "SMILES"])

            failures_file.write("SMILES\tReason\n")
            
            smiles_iterator = (line.strip() for line in input_file if line.strip())
            results_iterator = executor.map(_worker_convert_smiles, smiles_iterator, chunksize=chunksize)
            
            total_lines = 0
            start_time = time.time()
            self._update_progress(0, total_to_process, start_time, prefix="Converting")
            for i, (original_smiles, processed_smiles, selfies_result, error_message) in enumerate(results_iterator):
                total_lines += 1
                if selfies_result and processed_smiles not in processed_smiles_set:
                    writer.writerow([selfies_result, processed_smiles])
                    processed_smiles_set.add(processed_smiles)
                    successful_conversions += 1
                else:
                    if error_message is None and processed_smiles in processed_smiles_set:
                        error_message = "Duplicate SMILES"

                    failures_file.write(f"{original_smiles}\t{error_message}\n")
                    failed_conversions += 1
                    reason = error_message or "Unknown error"
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                
                # Update progress periodically
                if (i + 1) % (chunksize // 2 or 1) == 0:
                    self._update_progress(i + 1, total_to_process, start_time, prefix="Converting")
        
        # Final progress update
        self._update_progress(total_to_process, total_to_process, start_time, prefix="Converting")
        sys.stderr.write('\n')
        
        conversion_stats = {
            "total_processed": total_lines,
            "successful_conversions": successful_conversions,
            "duplicate_smiles": len(processed_smiles_set) - successful_conversions if successful_conversions < len(processed_smiles_set) else 0,
            "failed_conversions": failed_conversions,
            "success_rate": successful_conversions / total_lines if total_lines > 0 else 0,
            "failure_reasons": failure_reasons,
        }
        
        import json
        with open(stats_filepath, 'w') as stats_file:
            json.dump(conversion_stats, stats_file, indent=2)
        
        logger.info(f"Conversion complete. Success: {successful_conversions:,}, Failed: {failed_conversions:,} ({conversion_stats['success_rate']:.2%})")
        if failed_conversions > 0:
            logger.warning(f"Failed conversion details logged to: {failures_filepath}")
            logger.info(f"Aggregated statistics saved to: {stats_filepath}")
            
        return successful_conversions, failed_conversions
    
    def sample_random_smiles(
        self,
        target_count: int = 10_000_000,
        output_filename: str = "random_smiles.txt",
        max_files_to_process: Optional[int] = None,
        convert_to_selfies: bool = False,
        selfies_filename: Optional[str] = None,
        num_workers: int = 1
    ) -> str:
        """
        Download and sample random SMILES from PubChem via FTP.
        
        Note: This method is legacy. For faster and more memory-efficient processing,
        the 'usearch' method is recommended if the data is available.
        
        Args:
            target_count: Number of random SMILES to collect.
            output_filename: Name of the output file for sampled SMILES.
            max_files_to_process: Limit the number of SDF files to process.
            convert_to_selfies: If True, convert SMILES to SELFIES after collection.
            selfies_filename: Optional name for the SELFIES output file.
            num_workers: Number of workers for SELFIES conversion.
            
        Returns:
            Path to the output file containing sampled SMILES.
        """
        logger.info(f"Starting random SMILES sampling from FTP (target: {target_count:,})")
        
        file_pairs = self.get_available_files()
        random.shuffle(file_pairs)
        
        if max_files_to_process:
            file_pairs = file_pairs[:max_files_to_process]
            logger.info(f"Processing a random subset of {len(file_pairs)} files.")
        
        output_path = self.output_dir / output_filename
        collected_smiles = set()
        start_time = time.time()
        
        try:
            with open(output_path, 'w') as output_file:
                for i, (sdf_file, md5_file) in enumerate(file_pairs):
                    if len(collected_smiles) >= target_count:
                        logger.info("Target SMILES count reached.")
                        break
                    
                    logger.info(f"Processing file {i+1}/{len(file_pairs)}: {sdf_file}")
                    
                    sdf_path = self.download_file_with_aria2c(sdf_file, md5_file)
                    if not sdf_path:
                        logger.warning(f"Skipping file {sdf_file} due to download failure.")
                        continue
                    
                    try:
                        extracted_count = 0
                        for smiles in self.extract_smiles_from_sdf(sdf_path):
                            if smiles not in collected_smiles:
                                collected_smiles.add(smiles)
                                output_file.write(f"{smiles}\n")
                                extracted_count += 1
                                if len(collected_smiles) >= target_count:
                                    break
                        
                        logger.info(f"Extracted {extracted_count:,} new unique SMILES from {sdf_file}")
                        self._update_progress(len(collected_smiles), target_count, start_time, prefix="FTP Sampling")
                        
                    except SDFParsingError as e:
                        logger.error(f"Failed to parse {sdf_file}: {e}")
                        continue
        
        except Exception as e:
            raise PubChemDownloadError(f"A critical error occurred during SMILES sampling: {e}")
        
        sys.stderr.write('\n')
        logger.info(f"Sampling complete. Collected {len(collected_smiles):,} unique SMILES.")
        logger.info(f"Output saved to: {output_path}")
        
        if convert_to_selfies:
            self._run_selfies_conversion(output_filename, selfies_filename, num_workers)
        
        return str(output_path)
    
    def cleanup_cache(self) -> None:
        """Remove all cached files to free up disk space."""
        try:
            if self.cache_dir.exists():
                logger.info(f"Cleaning up cache directory: {self.cache_dir}")
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache cleaned up successfully.")
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def get_cache_size(self) -> int:
        """Get total size of cached files in bytes."""
        try:
            return sum(f.stat().st_size for f in self.cache_dir.glob('**/*') if f.is_file())
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
            return 0

    def sample_smiles_from_usearch_parquet(
        self,
        target_count: int,
        output_filename: str,
        convert_to_selfies: bool,
        selfies_filename: Optional[str],
        num_workers: int = 1
    ) -> None:
        """
        Samples SMILES from pre-downloaded Parquet files (USearch dataset).

        This method is memory-efficient and reads files in batches. The final count
        of SMILES will be an approximation of target_count due to the probabilistic
        sampling method.
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("The 'pyarrow' library is required for this functionality.")

        parquet_dir = self.cache_dir
        files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")
        
        logger.info(f"Found {len(files)} Parquet files.")

        logger.info("Estimating total number of molecules...")
        total_molecules = sum(pq.read_metadata(file_path).num_rows for file_path in files)
        if total_molecules == 0:
            logger.error("Could not determine total number of molecules from metadata. Aborting.")
            return

        sampling_rate = target_count / total_molecules
        logger.info(f"Targeting ~{target_count:,} from {total_molecules:,} molecules (sampling rate: {sampling_rate:.4%})")

        output_path = self.output_dir / output_filename
        collected_count = 0
        start_time = time.time()

        with open(output_path, "w") as f_out:
            self._update_progress(0, target_count, start_time, prefix="Parquet Sampling")
            for file_path in files:
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=2**16, columns=['smiles']):
                        smiles_list = batch['smiles'].to_pylist()
                        for smiles in smiles_list:
                            if random.random() < sampling_rate:
                                f_out.write(smiles + '\n')
                                collected_count += 1
                        # Update progress after each batch for responsiveness
                        self._update_progress(collected_count, target_count, start_time, prefix="Parquet Sampling")
                except Exception as e:
                    logger.error(f"Failed to process Parquet file {file_path}: {e}")
        
        sys.stderr.write('\n')
        logger.info(f"Finished sampling. Total SMILES collected: {collected_count:,}")
        logger.info(f"Output saved to: {output_path}")

        if convert_to_selfies:
            self._run_selfies_conversion(output_filename, selfies_filename, num_workers)
    
    def _run_selfies_conversion(
        self, 
        input_filename: str, 
        selfies_filename: Optional[str], 
        num_workers: int
    ) -> None:
        """Helper to run the SELFIES conversion process."""
        if not SELFIES_AVAILABLE:
            logger.error("SELFIES conversion requested but dependencies are not available.")
            raise ValueError("SELFIES conversion is not available.")
        
        if selfies_filename is None:
            stem = Path(input_filename).stem
            selfies_filename = f"{stem}_selfies.csv"
        
        logger.info("Starting SELFIES conversion...")
        self.convert_smiles_to_selfies_file(
            input_filename=input_filename,
            output_filename=selfies_filename,
            num_workers=num_workers
        )
        selfies_path = self.output_dir / selfies_filename
        logger.info(f"SELFIES data saved to: {selfies_path}")


def main():
    """
    Main interface for the PubChem SMILES data preparation pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PubChem SMILES/SELFIES Data Preparation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default="usearch", 
        choices=["usearch", "ftp"], 
        help="Download method to use. 'usearch' is faster and memory-efficient."
    )
    parser.add_argument("--limit", type=int, default=10_000_000, help="Target number of SMILES molecules.")
    parser.add_argument("--output", type=str, default="training.csv", help="Output filename for SMILES.")
    parser.add_argument("--data-dir", type=str, default="data", help="Main data directory.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of SDF files to process (FTP method only).")
    parser.add_argument("--cleanup", action="store_true", help="Clean up cache directory after completion.")
    parser.add_argument("--no-selfies", action="store_true", help="Disable the final SELFIES conversion step.")
    parser.add_argument("--selfies-output", type=str, default=None, help="Custom output filename for SELFIES.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose logging.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes for SELFIES conversion.")
    
    args = parser.parse_args()
    
    pipeline = PubChemSMILESPipeline(
        data_dir=args.data_dir,
        debug=args.debug,
        aria2c_connections=args.workers 
    )
    
    try:
        if args.method == 'usearch':
            pipeline.sample_smiles_from_usearch_parquet(
                target_count=args.limit,
                output_filename=args.output,
                convert_to_selfies=not args.no_selfies,
                selfies_filename=args.selfies_output,
                num_workers=args.workers
            )
        else:
            pipeline.sample_random_smiles(
                target_count=args.limit,
                output_filename=args.output,
                max_files_to_process=args.max_files,
                convert_to_selfies=not args.no_selfies,
                selfies_filename=args.selfies_output,
                num_workers=args.workers
            )
        
        cache_size_mb = pipeline.get_cache_size() / (1024 * 1024)
        logger.info(f"Current cache size: {cache_size_mb:.1f} MB")
        
        if args.cleanup:
            pipeline.cleanup_cache()
            
    except (PubChemDownloadError, SDFParsingError, FileNotFoundError, ImportError) as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.debug)
        sys.exit(1)
    
    logger.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    exit(main())
