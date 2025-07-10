#!/bin/bash

# =============================================================================
# CrossDock2020 Protein-Ligand Dataset Download & Processing Script
# =============================================================================
#
# Robust bash wrapper for downloading and processing CrossDock2020 dataset.
# Features:
#   - Advanced command-line interface
#   - Resumable downloads via state management
#   - Process control (status checks, killing jobs)
#   - Dependency checks
#   - Checksum verification
#
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/prepare_crossdock.py"
LOG_DIR="$SCRIPT_DIR/logs"

# State and Logging
STATE_FILE="$LOG_DIR/crossdock_state.json"
PROGRESS_LOG="$LOG_DIR/crossdock_progress.log"
ERROR_LOG="$LOG_DIR/crossdock_errors.log"

# CrossDock2020 Dataset URLs
CROSSDOCK_BASE_URL="http://bits.csb.pitt.edu/files/crossdock2020"
STRUCTURES_FILE="CrossDocked2020_v1.3.tgz"
TYPES_FILE="CrossDocked2020_v1.3_types.tgz"

# Default Parameters
OUTPUT_CSV="protein_ligand_pairs.csv"
DATA_DIR="data"
CLEANUP=false
RESUME=true
DEBUG_MODE=false
WORKERS=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
EXTRACT_TYPES=true

# --- UI Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Global State ---
START_TIME=$(date +%s)

# =============================================================================
# Logging and Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROGRESS_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROGRESS_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROGRESS_LOG" | tee -a "$ERROR_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROGRESS_LOG"
}

show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                CrossDock2020 Protein-Ligand Dataset Pipeline                 ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${BLUE}Features: Robust downloads • Data processing • CSV pair generation${NC}"
    echo -e "${BLUE}Output: Protein-ligand pairs → $OUTPUT_CSV${NC}"
    echo ""
}

show_usage() {
    cat <<EOF
${CYAN}CrossDock2020 Protein-Ligand Dataset Downloader & Processor${NC}
Usage: $0 [OPTIONS]

${GREEN}Basic Options:${NC}
  -o, --output OUTPUT      Output CSV filename (default: $OUTPUT_CSV)
  -d, --data-dir DIR       Main data directory (default: $DATA_DIR)

${GREEN}Processing Options:${NC}
  --workers NUM            Number of parallel workers for processing (default: $WORKERS)
  --no-extract-types       Skip extracting and processing types file
  
${GREEN}Advanced & Process Control:${NC}
  -r, --resume             Resume previous job (default: enabled)
  -R, --no-resume          Start fresh, ignoring previous state
  -C, --cleanup            Clean up cache directory after completion
  -s, --status             Show current download status
  -k, --kill               Kill the running download process
  --debug                  Enable debug mode for more verbose logging
  -h, --help               Show this help message

${YELLOW}Examples:${NC}
  $0                                # Download and process CrossDock2020 dataset
  $0 -o my_pairs.csv                # Custom output filename
  $0 --no-extract-types             # Skip types file processing
  $0 -s                             # Check current job status
  $0 -k                             # Stop the running job
EOF
}

# =============================================================================
# System and Dependency Checks
# =============================================================================

check_dependencies() {
    log_info "Checking system dependencies..."
    local missing_deps=()
    local missing_python_deps=()
    
    # Check system tools
    command -v python3 &> /dev/null || missing_deps+=("python3")
    command -v aria2c &> /dev/null || missing_deps+=("aria2c")
    command -v tar &> /dev/null || missing_deps+=("tar")
    
    # Check Python packages
    if command -v python3 &> /dev/null; then
        log_info "Checking Python package dependencies..."
        
        # Check for RDKit
        if ! python3 -c "import rdkit" &> /dev/null; then
            missing_python_deps+=("rdkit")
        fi
        

        
        # Check for other essential packages
        if ! python3 -c "import pandas" &> /dev/null; then
            missing_python_deps+=("pandas")
        fi
    fi
    
    # Report missing system dependencies
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing system dependencies: ${missing_deps[*]}"
        echo -e "${RED}Please install missing system dependencies and try again.${NC}"
        echo "Installation suggestions:"
        echo "  - aria2c: sudo apt-get install aria2 (Ubuntu/Debian) or brew install aria2 (macOS)"
        echo "  - tar: usually pre-installed on Unix systems"
        exit 1
    fi
    
    # Report missing Python dependencies
    if [ ${#missing_python_deps[@]} -ne 0 ]; then
        log_error "Missing Python dependencies: ${missing_python_deps[*]}"
        echo -e "${RED}Critical Python packages are missing!${NC}"
        echo ""
        echo -e "${YELLOW}Required installations:${NC}"
        
        for dep in "${missing_python_deps[@]}"; do
            case "$dep" in
                "rdkit")
                    echo -e "${CYAN}For RDKit:${NC}"
                    echo "  pip install rdkit"
                    echo "  OR: conda install -c conda-forge rdkit"
                    echo ""
                    ;;
                "pandas")
                    echo -e "${CYAN}For pandas:${NC}"
                    echo "  pip install pandas"
                    echo ""
                    ;;
            esac
        done
        echo "Please install the missing packages and run the script again."
        exit 1
    fi
    
    log_success "All dependencies satisfied."
}

check_disk_space() {
    local target_dir="$1"
    local required_gb="$2"
    log_info "Checking disk space in '$target_dir' (requires ~${required_gb}GB)..."
    
    local available_gb
    available_gb=$(df -BG "$target_dir" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        log_warn "Low disk space: ${available_gb}GB available, but ${required_gb}GB is recommended."
        read -p "Continue anyway? (y/N) " -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Download cancelled by user."
            exit 0
        fi
    fi
    log_success "Sufficient disk space available (${available_gb}GB)."
}

# =============================================================================
# State Management
# =============================================================================

save_state() {
    local pid="$1"; shift
    mkdir -p "$LOG_DIR"
    
    jq -n \
        --arg output "$OUTPUT_CSV" \
        --arg data_dir "$DATA_DIR" \
        --arg extract_types "$EXTRACT_TYPES" \
        --arg pid "$pid" \
        --arg start_time "$START_TIME" \
        --arg status "running" \
        --arg workers "$WORKERS" \
        '{output: $output, data_dir: $data_dir, extract_types: $extract_types, pid: $pid, start_time: $start_time, status: $status, workers: $workers}' \
        > "$STATE_FILE"
}

update_state_status() {
    local status="$1"
    if [ -f "$STATE_FILE" ]; then
        if command -v jq &> /dev/null; then
            jq ".status = \"$status\"" "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
        else
            sed -i "s/\"status\": \"[^\"]*\"/\"status\": \"$status\"/" "$STATE_FILE"
        fi
    fi
}

# =============================================================================
# Download Functions
# =============================================================================

download_file_with_aria2c() {
    local url="$1"
    local filename="$2"
    local target_dir="$3"
    local max_retries=3
    
    local filepath="$target_dir/$filename"
    
    # Check if download is complete by looking for aria2 control file
    # If .aria2 file exists, download was incomplete
    if [ -f "$filepath" ] && [ ! -f "$filepath.aria2" ]; then
        log_info "File '$filename' already exists and appears complete, skipping download."
        return 0
    elif [ -f "$filepath.aria2" ]; then
        log_info "Resuming incomplete download of '$filename'..."
    fi
    
    log_info "Downloading $filename..."
    local cmd=(
        aria2c
        --dir="$target_dir"
        --out="$filename"
        --max-tries="$max_retries"
        --retry-wait=1
        --timeout=600
        --connect-timeout=30
        --continue=true
        --max-connection-per-server=16
        --split=16
        --min-split-size=1M
        --max-download-limit=0
        --disable-ipv6=false
        --enable-http-pipelining=true
        --http-accept-gzip=true
        --reuse-uri=true
        --max-concurrent-downloads=5
        "$url"
    )
    
    if [ "$DEBUG_MODE" = true ]; then
        cmd+=(--summary-interval=1)
    fi
    
    if "${cmd[@]}"; then
        log_success "Successfully downloaded $filename"
        return 0
    else
        log_error "Failed to download $filename"
        return 1
    fi
}

download_crossdock_files() {
    local cache_dir="$1"
    mkdir -p "$cache_dir"
    
    log_info "Downloading CrossDock2020 dataset files..."
    
    # Download main structures archive
    if ! download_file_with_aria2c "$CROSSDOCK_BASE_URL/$STRUCTURES_FILE" "$STRUCTURES_FILE" "$cache_dir"; then
        log_error "Failed to download structures archive"
        return 1
    fi
    
    # Extract structures archive to crossdocked folder
    local extract_dir="$cache_dir/crossdocked"
    mkdir -p "$extract_dir"
    log_info "Extracting structures archive to $extract_dir (this may take a while)..."
    if tar -xzf "$cache_dir/$STRUCTURES_FILE" -C "$extract_dir"; then
        log_success "Structures archive extracted successfully to crossdocked/"
    else
        log_error "Failed to extract structures archive"
        return 1
    fi
    
    # Download types file if requested
    if [ "$EXTRACT_TYPES" = true ]; then
        if ! download_file_with_aria2c "$CROSSDOCK_BASE_URL/$TYPES_FILE" "$TYPES_FILE" "$cache_dir"; then
            log_error "Failed to download types file"
            return 1
        fi
        
        # Extract types file
        log_info "Extracting types file..."
        mkdir -p "$cache_dir/types"
        if tar -xzf "$cache_dir/$TYPES_FILE" -C "$cache_dir/types"; then
            log_success "Types file extracted successfully"
        else
            log_error "Failed to extract types file"
            return 1
        fi
    fi
    
    log_success "All CrossDock2020 files downloaded and extracted successfully"
    return 0
}

# =============================================================================
# Job Monitoring and Control
# =============================================================================

monitor_pipeline() {
    local pid=$1
    log_info "Monitoring pipeline process (PID: $pid). The script will show its own progress."
    log_info "Logs are being written to: $PROGRESS_LOG"
    
    wait "$pid"
    return $?
}

show_completion_stats() {
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    # Try to get final count from logs
    local total_pairs
    total_pairs=$(grep -o "Generated [0-9,]* protein-ligand pairs" "$PROGRESS_LOG" | tail -1 | grep -o "[0-9,]*" | tr -d ',' || echo 0)

    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                             PIPELINE COMPLETED                             ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    printf "║ Protein-Ligand Pairs:   %'12d                                      ║\n" "$total_pairs"
    printf "║ Total Duration:          %02d:%02d:%02d                                           ║\n" "$hours" "$minutes" "$seconds"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║ Output CSV: $OUTPUT_CSV                                                      ║"
    echo "║ Full logs are available in: $LOG_DIR                                 ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# =============================================================================
# Main Pipeline Execution
# =============================================================================

start_pipeline() {
    if [ "$RESUME" = false ]; then
        log_info "Preparing for new pipeline run by cleaning logs and state files (--no-resume)."
        
        # Clean state and logs
        rm -f "$STATE_FILE"
        find "$LOG_DIR" -mindepth 1 -delete 2>/dev/null || true
        
        local output_dir="$DATA_DIR/output"
        if [ -d "$output_dir" ]; then
            rm -f "$output_dir/positive_pairs.csv" "$output_dir/negative_pairs.csv"
        fi
        
        log_info "Previous run artifacts have been cleared."
    else
        log_info "Starting or resuming pipeline. Existing files will be preserved."
    fi

    START_TIME=$(date +%s)

    echo -e "${GREEN}Starting CrossDock2020 Dataset Pipeline${NC}"
    echo "--------------------------------------------------"
    log_info "Output CSV file: $OUTPUT_CSV"
    log_info "Workers: $WORKERS"
    log_info "Extract types file: $EXTRACT_TYPES"
    
    local data_cache_dir="$DATA_DIR/crossdocked"
    check_disk_space "$data_cache_dir" 10  # CrossDock2020 is smaller than PubChem
    
    # Download files
    if ! download_crossdock_files "$data_cache_dir"; then
        log_error "Failed to download CrossDock2020 files"
        exit 1
    fi
    
    # Construct Python command
    local python_cmd=(
        python3 "$PYTHON_SCRIPT"
        --output "$OUTPUT_CSV"
        --data-dir "$DATA_DIR"
        --workers "$WORKERS"
        --debug  # Enable debug mode by default for troubleshooting
    )
    [ "$CLEANUP" = true ] && python_cmd+=(--cleanup)
    [ "$EXTRACT_TYPES" = false ] && python_cmd+=(--no-extract-types)
    
    # Execute Python pipeline in the background and monitor it
    "${python_cmd[@]}" 2>&1 | tee -a "$PROGRESS_LOG" &
    local pid=$!
    
    save_state "$pid"
    
    monitor_pipeline "$pid"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Pipeline completed successfully!"
        update_state_status "completed"
        show_completion_stats
    else
        log_error "Pipeline failed with exit code: $exit_code"
        update_state_status "failed"
        return $exit_code
    fi
}

# =============================================================================
# Status and Control Commands
# =============================================================================

show_status() {
    if [ ! -f "$STATE_FILE" ]; then
        echo -e "${YELLOW}No pipeline state file found. No job appears to be running or have run.${NC}"
        return 0
    fi
    
    echo -e "${CYAN}Current Pipeline Status:${NC}"
    echo "========================="
    
    if ! command -v jq &> /dev/null; then
        log_warn "jq is not installed. Displaying raw state file."
        cat "$STATE_FILE"
        return 0
    fi
    
    local status; status=$(jq -r '.status // "unknown"' "$STATE_FILE")
    local pid; pid=$(jq -r '.pid // "unknown"' "$STATE_FILE")
    
    jq -r '"Status: \(.status)\nProcess ID: \(.pid)\nWorkers: \(.workers)\nOutput: \(.output)"' "$STATE_FILE"
    
    if [ "$pid" != "unknown" ] && kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}Process is currently running.${NC}"
        local start_time; start_time=$(jq -r '.start_time // 0' "$STATE_FILE")
        local duration=$(( $(date +%s) - start_time ))
        echo "Running time: $(date -u -d "@$duration" +'%Hh %Mm %Ss')"
    else
        echo -e "${RED}Process is not running.${NC}"
    fi
    
    if [ -f "$PROGRESS_LOG" ]; then
        echo -e "\n${CYAN}Recent Progress Log:${NC}"
        echo "====================="
        tail -n 10 "$PROGRESS_LOG"
    fi
}

kill_pipeline() {
    if [ ! -f "$STATE_FILE" ]; then
        log_warn "No state file found. Cannot determine process to kill."
        return 1
    fi

    if ! command -v jq &> /dev/null; then
        log_error "Cannot kill process - jq is not installed to read PID from state file."
        return 1
    fi
    
    local pid; pid=$(jq -r '.pid // ""' "$STATE_FILE")
    
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log_info "Killing pipeline process (PID: $pid) and its children..."
        kill -TERM -"$pid" 2>/dev/null || true
        sleep 2
        
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "Process still running, sending SIGKILL."
            kill -KILL -"$pid" 2>/dev/null || true
        fi
        
        update_state_status "killed"
        log_success "Pipeline process terminated."
    else 
        log_warn "No running pipeline process found to kill."
    fi
}

cleanup_on_exit() {
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        log_warn "Script interrupted. Cleaning up background process..."
        kill_pipeline
        update_state_status "interrupted"
    fi
}
trap cleanup_on_exit INT TERM

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    # Ensure jq is available for state management if possible
    if ! command -v jq &>/dev/null; then
        log_warn "jq command not found. State management will be limited."
    fi

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--output) OUTPUT_CSV="$2"; shift 2 ;;
            -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
            --workers) WORKERS="$2"; shift 2 ;;
            -r|--resume) RESUME=true; shift ;;
            -R|--no-resume) RESUME=false; shift ;;
            -C|--cleanup) CLEANUP=true; shift ;;
            --no-extract-types) EXTRACT_TYPES=false; shift ;;
            -s|--status) show_status; exit 0 ;;
            -k|--kill) kill_pipeline; exit 0 ;;
            --debug) DEBUG_MODE=true; shift ;;
            -h|--help) show_usage; exit 0 ;;
            *) log_error "Unknown option: $1"; show_usage; exit 1 ;;
        esac
    done
    
    mkdir -p "$LOG_DIR" "$DATA_DIR/crossdocked" "$DATA_DIR/output"
    
    show_banner
    check_dependencies
    
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_error "Python pipeline script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Handle resume logic
    if [ "$RESUME" = true ] && [ -f "$STATE_FILE" ]; then
        local status; status=$(jq -r '.status // "unknown"' "$STATE_FILE")
        local pid; pid=$(jq -r '.pid // ""' "$STATE_FILE")
        
        if [ "$status" = "running" ] && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log_warn "Pipeline is already running (PID: $pid)."
            read -p "Continue monitoring this job? (Y/n) " -r response
            if [[ ! "$response" =~ ^[Nn]$ ]]; then
                monitor_pipeline "$pid"
                exit $?
            else
                exit 0
            fi
        fi
    fi
    
    start_pipeline
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 