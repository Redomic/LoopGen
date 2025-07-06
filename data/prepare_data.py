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
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError
import time
import hashlib
import re

# Add molecule_utils to path
sys.path.append(str(Path(__file__).parent.parent / "molecule_utils"))

try:
    from convert import (
        convert_smiles_to_selfies,
        batch_convert_smiles_to_selfies,
        MolecularConversionError,
        InvalidSMILESError
    )
    SELFIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SELFIES conversion not available: {e}")
    SELFIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        cache_subdir: str = "cache",
        output_subdir: str = "output",
        aria2c_threads: int = 5,
        max_retries: int = 5,
        chunk_size: int = 8192,
        debug: bool = False
    ):
        """
        Initialize the PubChem SMILES pipeline.
        
        Args:
            data_dir: Main data directory (all data stored here)
            cache_subdir: Cache subdirectory within data_dir
            output_subdir: Output subdirectory within data_dir
            aria2c_threads: Number of threads for aria2c downloads
            max_retries: Maximum number of retry attempts for failed operations
            chunk_size: Chunk size for file operations
            debug: Enable debug mode for more verbose logging
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / cache_subdir
        self.output_dir = self.data_dir / output_subdir
        self.aria2c_threads = aria2c_threads
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.debug = debug
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup aria2c check
        self._check_aria2c()
        
        logger.info(f"Pipeline initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  SELFIES conversion: {'Available' if SELFIES_AVAILABLE else 'Not available'}")
    
    def _check_aria2c(self) -> None:
        """Check if aria2c is available and accessible."""
        try:
            result = subprocess.run(
                ["aria2c", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise PubChemDownloadError("aria2c is not working properly")
            logger.info("aria2c is available and ready")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise PubChemDownloadError(
                "aria2c is not installed or not accessible. "
                "Please install aria2c: sudo apt-get install aria2 (Ubuntu/Debian) "
                "or brew install aria2 (macOS)"
            )
    
    def get_available_files(self) -> List[Tuple[str, str]]:
        """
        Get list of available SDF files from PubChem FTP.
        
        Returns:
            List of tuples (filename, md5_filename) for available SDF files
            
        Raises:
            PubChemDownloadError: If unable to fetch file list
        """
        logger.info("Fetching available files from PubChem FTP...")
        
        try:
            request = Request(self.BASE_URL)
            request.add_header('User-Agent', 'PubChem-Pipeline/1.0')
            
            with urlopen(request, timeout=30) as response:
                content = response.read().decode('utf-8')
            
            # Parse HTML to extract SDF file names
            sdf_pattern = r'href="([^"]*\.sdf\.gz)"'
            md5_pattern = r'href="([^"]*\.sdf\.gz\.md5)"'
            
            sdf_files = re.findall(sdf_pattern, content)
            md5_files = re.findall(md5_pattern, content)
            
            # Match SDF files with their MD5 checksums
            file_pairs = []
            for sdf_file in sdf_files:
                md5_file = sdf_file + '.md5'
                if md5_file in md5_files:
                    file_pairs.append((sdf_file, md5_file))
            
            logger.info(f"Found {len(file_pairs)} SDF files available for download")
            return file_pairs
            
        except URLError as e:
            raise PubChemDownloadError(f"Failed to fetch file list: {e}")
        except Exception as e:
            raise PubChemDownloadError(f"Unexpected error fetching file list: {e}")
    
    def download_file_with_aria2c(self, filename: str, md5_filename: str) -> bool:
        """
        Download a single SDF file using aria2c with checksum verification.
        
        Args:
            filename: Name of the SDF file to download
            md5_filename: Name of the corresponding MD5 file
            
        Returns:
            True if download successful, False otherwise
        """
        sdf_url = f"{self.BASE_URL}{filename}"
        md5_url = f"{self.BASE_URL}{md5_filename}"
        
        sdf_path = self.cache_dir / filename
        md5_path = self.cache_dir / md5_filename
        
        # Skip if file already exists and is valid
        if sdf_path.exists() and self._verify_file_integrity(sdf_path, md5_path):
            logger.info(f"File {filename} already exists and is valid, skipping download")
            return True
        
        try:
            # Download MD5 file first
            logger.info(f"Downloading MD5 checksum for {filename}")
            aria2c_cmd_md5 = [
                "aria2c",
                "-x", "1", "-s", "1",  # Use single connection for small file
                "--dir", str(self.cache_dir),
                "--out", md5_filename,
                "--retry-wait", "3",
                "--max-tries", str(self.max_retries),
                "--timeout", "60",
                "--connect-timeout", "30",
                md5_url
            ]
            
            result_md5 = subprocess.run(aria2c_cmd_md5, capture_output=True, text=True)
            if result_md5.returncode != 0:
                logger.error(f"Failed to download MD5 for {filename}: {result_md5.stderr}")
                return False
            
            # Get file size for logging
            file_size = self._get_remote_file_size(sdf_url)
            file_size_mb = file_size / (1024 * 1024) if file_size else 0
            
            # Download SDF file with progress bar
            logger.info(f"Downloading {filename} ({file_size_mb:.1f} MB)")
            aria2c_cmd_sdf = [
                "aria2c",
                "-x", str(self.aria2c_threads),
                "-s", str(self.aria2c_threads),
                "--min-split-size", "10M",
                "--dir", str(self.cache_dir),
                "--out", filename,
                "--retry-wait", "3",
                "--max-tries", str(self.max_retries),
                "--timeout", "300",
                "--connect-timeout", "30",
                sdf_url
            ]
            
            if self.debug:
                aria2c_cmd_sdf.extend(["--summary-interval", "1"])

            result_sdf = subprocess.run(aria2c_cmd_sdf)
            
            if result_sdf.returncode != 0:
                logger.error(f"Failed to download {filename}")
                return False

            # Verify integrity
            logger.info(f"Verifying integrity of {filename}...")
            if self._verify_file_integrity(sdf_path, md5_path):
                logger.info(f"Successfully downloaded and verified {filename}")
                return True
            else:
                logger.error(f"Integrity check failed for {filename}")
                # Clean up corrupted file
                if sdf_path.exists():
                    sdf_path.unlink()
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False
    
    def _get_remote_file_size(self, url: str) -> Optional[int]:
        """
        Get the size of a remote file without downloading it.
        
        Args:
            url: URL of the file
            
        Returns:
            File size in bytes, or None if unable to determine
        """
        try:
            request = Request(url)
            request.get_method = lambda: 'HEAD'
            response = urlopen(request, timeout=10)
            return int(response.headers.get('Content-Length', 0))
        except Exception:
            return None
    
    def _verify_file_integrity(self, file_path: Path, md5_path: Path) -> bool:
        """
        Verify file integrity using MD5 checksum.
        
        Args:
            file_path: Path to the file to verify
            md5_path: Path to the MD5 checksum file
            
        Returns:
            True if integrity check passes, False otherwise
        """
        try:
            if not file_path.exists() or not md5_path.exists():
                return False
            
            # Read expected MD5
            with open(md5_path, 'r') as f:
                expected_md5 = f.read().strip().split()[0]
            
            # Calculate actual MD5
            actual_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(self.chunk_size), b''):
                    actual_md5.update(chunk)
            
            return actual_md5.hexdigest() == expected_md5
            
        except Exception as e:
            logger.error(f"Error verifying integrity: {e}")
            return False
    
    def extract_smiles_from_sdf(self, sdf_path: Path) -> Iterator[str]:
        """
        Extract SMILES strings from a compressed SDF file.
        
        Args:
            sdf_path: Path to the compressed SDF file
            
        Yields:
            SMILES strings found in the SDF file
            
        Raises:
            SDFParsingError: If unable to parse SDF file
        """
        try:
            with gzip.open(sdf_path, 'rt', encoding='utf-8') as f:
                current_compound = []
                
                for line in f:
                    line = line.strip()
                    
                    if line == '$$$$':  # End of compound record
                        if current_compound:
                            smiles = self._extract_smiles_from_compound(current_compound)
                            if smiles:
                                yield smiles
                        current_compound = []
                    else:
                        current_compound.append(line)
                
                # Handle last compound if file doesn't end with $$$$
                if current_compound:
                    smiles = self._extract_smiles_from_compound(current_compound)
                    if smiles:
                        yield smiles
                        
        except Exception as e:
            raise SDFParsingError(f"Error parsing SDF file {sdf_path}: {e}")
    
    def _extract_smiles_from_compound(self, compound_lines: List[str]) -> Optional[str]:
        """
        Extract SMILES from a single compound's SDF record.
        
        Args:
            compound_lines: List of lines for a single compound
            
        Returns:
            SMILES string if found, None otherwise
        """
        try:
            # Look for SMILES in property blocks
            for i, line in enumerate(compound_lines):
                if 'PUBCHEM_OPENEYE_CAN_SMILES' in line or 'PUBCHEM_CANONICAL_SMILES' in line:
                    # SMILES should be on the next line
                    if i + 1 < len(compound_lines):
                        smiles = compound_lines[i + 1].strip()
                        if smiles and self._is_valid_smiles_format(smiles):
                            return smiles
            
            return None
            
        except Exception:
            return None
    
    def _is_valid_smiles_format(self, smiles: str) -> bool:
        """
        Basic validation of SMILES format.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if format appears valid, False otherwise
        """
        if not smiles or len(smiles) < 2:
            return False
        
        # Basic checks for valid SMILES characters
        valid_chars = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[]{}=#@+-/\\.')
        return all(c in valid_chars for c in smiles)
    
    def convert_smiles_to_selfies_file(
        self,
        smiles_file: str,
        selfies_file: str,
        batch_size: int = 10000
    ) -> Tuple[int, int]:
        """
        Convert a file of SMILES to SELFIES format.
        
        Args:
            smiles_file: Path to input SMILES file
            selfies_file: Path to output SELFIES file
            batch_size: Number of SMILES to process in each batch
            
        Returns:
            Tuple of (successful_conversions, failed_conversions)
            
        Raises:
            ValueError: If SELFIES conversion is not available
        """
        if not SELFIES_AVAILABLE:
            raise ValueError("SELFIES conversion is not available. Please install required packages.")
        
        smiles_path = self.output_dir / smiles_file
        selfies_path = self.output_dir / selfies_file
        
        if not smiles_path.exists():
            raise FileNotFoundError(f"SMILES file not found: {smiles_path}")
        
        logger.info(f"Converting SMILES to SELFIES: {smiles_file} -> {selfies_file}")
        
        successful = 0
        failed = 0
        
        with open(smiles_path, 'r') as input_file, open(selfies_path, 'w') as output_file:
            batch = []
            
            for line_num, line in enumerate(input_file, 1):
                smiles = line.strip()
                if not smiles:
                    continue
                
                batch.append(smiles)
                
                if len(batch) >= batch_size:
                    batch_successful, batch_failed = self._process_smiles_batch(batch, output_file)
                    successful += batch_successful
                    failed += batch_failed
                    batch = []
                    
                    if line_num % (batch_size * 10) == 0:
                        logger.info(f"Processed {line_num:,} lines. Success: {successful:,}, Failed: {failed:,}")
            
            # Process remaining batch
            if batch:
                batch_successful, batch_failed = self._process_smiles_batch(batch, output_file)
                successful += batch_successful
                failed += batch_failed
        
        logger.info(f"Conversion completed. Success: {successful:,}, Failed: {failed:,}")
        return successful, failed
    
    def _process_smiles_batch(self, smiles_batch: List[str], output_file) -> Tuple[int, int]:
        """
        Process a batch of SMILES for conversion to SELFIES.
        
        Args:
            smiles_batch: List of SMILES strings to convert
            output_file: Output file handle for writing SELFIES
            
        Returns:
            Tuple of (successful_conversions, failed_conversions)
        """
        try:
            selfies_batch = batch_convert_smiles_to_selfies(
                smiles_batch,
                validate_input=True,
                skip_invalid=True
            )
            
            successful = 0
            failed = 0
            
            for smiles, selfies in zip(smiles_batch, selfies_batch):
                if selfies is not None:
                    output_file.write(f"{selfies}\n")
                    successful += 1
                else:
                    failed += 1
            
            return successful, failed
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return 0, len(smiles_batch)
    
    def sample_random_smiles(
        self,
        target_count: int = 10_000_000,
        output_filename: str = "random_smiles.txt",
        max_files_to_process: Optional[int] = None,
        convert_to_selfies: bool = False,
        selfies_filename: Optional[str] = None
    ) -> str:
        """
        Download and sample random SMILES from PubChem.
        
        Args:
            target_count: Number of random SMILES to collect
            output_filename: Name of output file
            max_files_to_process: Maximum number of SDF files to process (None for all)
            convert_to_selfies: Whether to convert SMILES to SELFIES after collection
            selfies_filename: Name of SELFIES output file (auto-generated if None)
            
        Returns:
            Path to the output file containing sampled SMILES
            
        Raises:
            PubChemDownloadError: If download process fails
        """
        logger.info(f"Starting random SMILES sampling (target: {target_count:,})")
        
        # Get available files
        file_pairs = self.get_available_files()
        
        if max_files_to_process:
            file_pairs = file_pairs[:max_files_to_process]
            logger.info(f"Limited processing to {max_files_to_process} files")
        
        # Randomly shuffle files to ensure diverse sampling
        random.shuffle(file_pairs)
        
        output_path = self.output_dir / output_filename
        collected_smiles = set()  # Use set to avoid duplicates
        
        try:
            with open(output_path, 'w') as output_file:
                for i, (sdf_file, md5_file) in enumerate(file_pairs):
                    if len(collected_smiles) >= target_count:
                        break
                    
                    logger.info(f"Processing file {i+1}/{len(file_pairs)}: {sdf_file}")
                    
                    # Download file
                    if not self.download_file_with_aria2c(sdf_file, md5_file):
                        logger.warning(f"Failed to download {sdf_file}, skipping...")
                        continue
                    
                    # Extract SMILES
                    sdf_path = self.cache_dir / sdf_file
                    try:
                        file_smiles = list(self.extract_smiles_from_sdf(sdf_path))
                        logger.info(f"Extracted {len(file_smiles):,} SMILES from {sdf_file}")
                        
                        # Random sample from this file
                        if file_smiles:
                            # Sample with replacement to handle large target counts
                            needed = target_count - len(collected_smiles)
                            sample_size = min(needed, len(file_smiles))
                            
                            sampled = random.sample(file_smiles, sample_size)
                            
                            for smiles in sampled:
                                if smiles not in collected_smiles:
                                    collected_smiles.add(smiles)
                                    output_file.write(f"{smiles}\n")
                                    
                                    if len(collected_smiles) >= target_count:
                                        break
                        
                        logger.info(f"Total collected: {len(collected_smiles):,}/{target_count:,}")
                        
                    except SDFParsingError as e:
                        logger.error(f"Failed to parse {sdf_file}: {e}")
                        continue
                    
                    # TODO: Clean up downloaded file to save space
                    # sdf_path.unlink()
        
        except Exception as e:
            raise PubChemDownloadError(f"Error during SMILES sampling: {e}")
        
        logger.info(f"Completed! Collected {len(collected_smiles):,} unique SMILES")
        logger.info(f"Output saved to: {output_path}")
        
        # Convert to SELFIES if requested
        if convert_to_selfies:
            if not SELFIES_AVAILABLE:
                logger.error("SELFIES conversion requested but not available")
                raise ValueError("SELFIES conversion is not available")
            
            if selfies_filename is None:
                # Auto-generate SELFIES filename
                stem = Path(output_filename).stem
                selfies_filename = f"{stem}_selfies.txt"
            
            logger.info(f"Converting SMILES to SELFIES...")
            successful, failed = self.convert_smiles_to_selfies_file(
                output_filename,
                selfies_filename
            )
            
            selfies_path = self.output_dir / selfies_filename
            logger.info(f"SELFIES conversion completed:")
            logger.info(f"  Success: {successful:,}")
            logger.info(f"  Failed: {failed:,}")
            logger.info(f"  SELFIES output: {selfies_path}")
        
        return str(output_path)
    
    def cleanup_cache(self) -> None:
        """Remove all cached files to free up disk space."""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def get_cache_size(self) -> int:
        """
        Get total size of cached files in bytes.
        
        Returns:
            Total size of cache directory in bytes
        """
        total_size = 0
        try:
            for path in self.cache_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
        return total_size


def main():
    """
    Main interface for the PubChem SMILES pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="PubChem SMILES/SELFIES Data Preparation Pipeline")
    parser.add_argument("--limit", type=int, default=10_000_000, help="Target number of SMILES molecules")
    parser.add_argument("--output", type=str, default="training.txt", help="Output filename for SMILES")
    parser.add_argument("--data-dir", type=str, default="data", help="Main data directory")
    parser.add_argument("--cache-subdir", type=str, default="cache", help="Cache subdirectory")
    parser.add_argument("--output-subdir", type=str, default="output", help="Output subdirectory")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of SDF files to process")
    parser.add_argument("--cleanup", action="store_true", help="Clean up cache directory after completion")
    parser.add_argument("--no-convert-to-selfies", action="store_true", help="Disable SELFIES conversion")
    parser.add_argument("--selfies-output", type=str, default=None, help="Output filename for SELFIES")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose logging")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PubChemSMILESPipeline(
        data_dir=args.data_dir,
        cache_subdir=args.cache_subdir,
        output_subdir=args.output_subdir,
        debug=args.debug
    )
    
    try:
        # Run sampling
        output_path = pipeline.sample_random_smiles(
            target_count=args.limit,
            output_filename=args.output,
            max_files_to_process=args.max_files,
            convert_to_selfies=not args.no_convert_to_selfies,
            selfies_filename=args.selfies_output
        )
        
        print(f"Successfully collected {args.limit:,} SMILES")
        print(f"Output saved to: {output_path}")
        
        # Show cache size
        cache_size_mb = pipeline.get_cache_size() / (1024 * 1024)
        print(f"Cache size: {cache_size_mb:.1f} MB")
        
        # Cleanup if requested
        if args.cleanup:
            pipeline.cleanup_cache()
            print("Cache cleaned up")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
