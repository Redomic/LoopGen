#!/bin/bash

# =============================================================================
# PubChem SMILES/SELFIES Download & Processing Script
# =============================================================================
#
# Robust bash wrapper for the Python data preparation pipeline.
# Features:
#   - Advanced command-line interface
#   - Resumable downloads via state management
#   - Process control (status checks, killing jobs)
#   - Dependency checks
#
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/prepare_data.py"
LOG_DIR="$SCRIPT_DIR/logs"

# State and Logging
STATE_FILE="$LOG_DIR/download_state.json"
PROGRESS_LOG="$LOG_DIR/download_progress.log"
ERROR_LOG="$LOG_DIR/download_errors.log"

# Default Parameters
LIMIT=10000000
OUTPUT_SMILES="training.txt"
DATA_DIR="data"
MAX_FILES=""
CLEANUP=false
RESUME=true
CONVERT_TO_SELFIES=true
OUTPUT_SELFIES=""
DEBUG_MODE=false
DOWNLOAD_METHOD="usearch"
WORKERS=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# --- UI Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Global State ---
TOTAL_SMILES=0
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
    echo "║                PubChem SMILES/SELFIES Data Preparation Pipeline              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${BLUE}Features: Multi-threaded processing • Progress tracking • SELFIES conversion${NC}"
    echo -e "${BLUE}Default: $(printf "%'d" $LIMIT) molecules → $OUTPUT_SMILES + ${OUTPUT_SMILES%.*}_selfies.txt${NC}"
    echo ""
}

show_usage() {
    cat <<EOF
${CYAN}PubChem SMILES/SELFIES Downloader & Processor${NC}
Usage: $0 [OPTIONS]

${GREEN}Basic Options:${NC}
  -l, --limit LIMIT        Number of SMILES to collect (default: $(printf "%'d" $LIMIT))
  -o, --output OUTPUT      Output filename for SMILES (default: $OUTPUT_SMILES)
  -d, --data-dir DIR       Main data directory (default: $DATA_DIR)

${GREEN}Download Method:${NC}
  --usearch                Use AWS S3 Parquet files for fast, memory-efficient processing (recommended)
  --ftp                    Use legacy FTP download method (slower, less reliable)

${GREEN}Conversion Options:${NC}
  --workers NUM            Number of parallel workers for conversion (default: $WORKERS)
  --no-selfies             Disable SELFIES conversion (enabled by default)
  --selfies-output NAME    Set a custom filename for the SELFIES output

${GREEN}Advanced & Process Control:${NC}
  -m, --max-files NUM      Max FTP files to process (FTP mode only, default: all)
  -r, --resume             Resume previous job (default: enabled)
  -R, --no-resume          Start fresh, ignoring previous state
  -C, --cleanup            Clean up cache directory after completion
  -s, --status             Show current download status
  -k, --kill               Kill the running download process
  --debug                  Enable debug mode for more verbose logging
  -h, --help               Show this help message

${YELLOW}Examples:${NC}
  $0                                # Download 10M molecules, convert to SELFIES using usearch method
  $0 -l 5000000 --ftp               # Download 5M molecules using legacy FTP
  $0 --no-selfies                   # Download SMILES only (no SELFIES conversion)
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
    
    command -v python3 &> /dev/null || missing_deps+=("python3")
    
    if [ "$DOWNLOAD_METHOD" = "usearch" ]; then
        command -v aws &> /dev/null || missing_deps+=("aws-cli")
        python3 -c "import pyarrow" &> /dev/null || missing_deps+=("python3-pyarrow")
    else # ftp
        command -v aria2c &> /dev/null || missing_deps+=("aria2c")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        echo -e "${RED}Please install missing dependencies and try again.${NC}"
        echo "Suggestions:"
        echo "  - For python3-pyarrow: pip install pyarrow"
        echo "  - For aws-cli: pip install awscli"
        echo "  - For others: sudo apt-get install -y <package> or brew install <package>"
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
    
    # Store all passed arguments in the state file
    jq -n \
        --arg limit "$LIMIT" \
        --arg output "$OUTPUT_SMILES" \
        --arg data_dir "$DATA_DIR" \
        --arg max_files "$MAX_FILES" \
        --arg convert_to_selfies "$CONVERT_TO_SELFIES" \
        --arg selfies_output "$OUTPUT_SELFIES" \
        --arg pid "$pid" \
        --arg start_time "$START_TIME" \
        --arg status "running" \
        --arg workers "$WORKERS" \
        '{limit: $limit, output: $output, data_dir: $data_dir, max_files: $max_files, convert_to_selfies: $convert_to_selfies, selfies_output: $selfies_output, pid: $pid, start_time: $start_time, status: $status, workers: $workers}' \
        > "$STATE_FILE"
}

update_state_status() {
    local status="$1"
    if [ -f "$STATE_FILE" ]; then
        # Use jq to update status if available, otherwise use sed
        if command -v jq &> /dev/null; then
            jq ".status = \"$status\"" "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
        else
            sed -i "s/\"status\": \"[^\"]*\"/\"status\": \"$status\"/" "$STATE_FILE"
        fi
    fi
}

# =============================================================================
# Job Monitoring and Control
# =============================================================================

monitor_pipeline() {
    local pid=$1
    log_info "Monitoring pipeline process (PID: $pid). The script will show its own progress."
    log_info "Logs are being written to: $PROGRESS_LOG"
    
    # Wait for the background process to complete
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
    TOTAL_SMILES=$(grep -o "Total SMILES collected: [0-9,]*" "$PROGRESS_LOG" | tail -1 | grep -o "[0-9,]*" | tr -d ',' || echo 0)
    
    local avg_rate=0
    if [ "$duration" -gt 0 ]; then
        avg_rate=$((TOTAL_SMILES * 3600 / duration))
    fi
    
    local selfies_file="${OUTPUT_SMILES%.*}_selfies.txt"
    local selfies_info="SELFIES file: $selfies_file"
    if [ "$CONVERT_TO_SELFIES" = false ]; then
        selfies_info="SELFIES conversion was disabled."
    fi

    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                             PIPELINE COMPLETED                             ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    printf "║ Molecules Collected: %'12d                                      ║\n" "$TOTAL_SMILES"
    printf "║ Total Duration:      %02d:%02d:%02d                                           ║\n" "$hours" "$minutes" "$seconds"
    printf "║ Average Rate:        %'12.0f molecules/hour                         ║\n" "$avg_rate"
    printf "║ %-66s ║\n" "$selfies_info"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║ SMILES file: $OUTPUT_SMILES                                                   ║"
    echo "║ Full logs are available in: $LOG_DIR                                 ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# =============================================================================
# Main Pipeline Execution
# =============================================================================

start_pipeline() {
    log_info "Preparing for new pipeline run by cleaning logs, output, and state files."
    
    # Clean directories and state file safely to ensure a fresh start.
    rm -f "$STATE_FILE"
    # Use find to delete contents without deleting the directories themselves.
    # The '|| true' part ensures the script doesn't fail if a directory doesn't exist.
    find "$LOG_DIR" -mindepth 1 -delete 2>/dev/null || true
    
    local output_dir="$DATA_DIR/output"
    # Only try to clean output dir if it exists.
    if [ -d "$output_dir" ]; then
        find "$output_dir" -mindepth 1 -delete 2>/dev/null || true
    fi
    
    log_info "Previous run artifacts have been cleared."

    START_TIME=$(date +%s)

    echo -e "${GREEN}Starting PubChem Data Pipeline${NC}"
    echo "--------------------------------------------------"
    log_info "Download method: $DOWNLOAD_METHOD"
    log_info "Target molecules: $(printf "%'d" "$LIMIT")"
    log_info "SMILES output file: $OUTPUT_SMILES"
    
    if [ "$CONVERT_TO_SELFIES" = true ]; then
        local selfies_name="$OUTPUT_SELFIES"
        if [ -z "$selfies_name" ]; then
            selfies_name="${OUTPUT_SMILES%.*}_selfies.txt"
        fi
        log_info "SELFIES conversion: ENABLED -> $selfies_name"
    else
        log_info "SELFIES conversion: DISABLED"
    fi
    
    local data_cache_dir="$DATA_DIR/cache"
    if [ "$DOWNLOAD_METHOD" = "usearch" ]; then
        log_info "Data source: AWS S3 (usearch-molecules)"
        mkdir -p "$data_cache_dir"
        check_disk_space "$data_cache_dir" 35
        
        log_info "Syncing PubChem Parquet files from AWS S3... (this may take a while)"
        if ! aws s3 sync --no-sign-request "s3://usearch-molecules/data/pubchem/parquet/" "$data_cache_dir"; then
            log_error "Failed to download data from S3. Check your connection and AWS CLI setup."
            exit 1
        fi
        log_success "S3 sync complete."

    else # ftp
        check_disk_space "$data_cache_dir" 50
    fi
    
    # Construct Python command
    local python_cmd=(
        python3 "$PYTHON_SCRIPT"
        --method "$DOWNLOAD_METHOD"
        --limit "$LIMIT"
        --output "$OUTPUT_SMILES"
        --data-dir "$DATA_DIR"
        --workers "$WORKERS"
    )
    [ "$DEBUG_MODE" = true ] && python_cmd+=(--debug)
    [ -n "$MAX_FILES" ] && python_cmd+=(--max-files "$MAX_FILES")
    [ "$CLEANUP" = true ] && python_cmd+=(--cleanup)
    [ "$CONVERT_TO_SELFIES" = false ] && python_cmd+=(--no-selfies)
    [ -n "$OUTPUT_SELFIES" ] && python_cmd+=(--selfies-output "$OUTPUT_SELFIES")
    
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
    
    jq -r '"Status: \(.status)\nTarget SMILES: \(.limit|tonumber|tostring|gsub(",";"")|tonumber|tostring)\nProcess ID: \(.pid)\nWorkers: \(.workers)"' "$STATE_FILE"
    
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
        # Kill the entire process group to stop child processes (like aria2c)
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
            -l|--limit) LIMIT="$2"; shift 2 ;;
            -o|--output) OUTPUT_SMILES="$2"; shift 2 ;;
            -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
            -m|--max-files) MAX_FILES="$2"; shift 2 ;;
            --workers) WORKERS="$2"; shift 2 ;;
            -r|--resume) RESUME=true; shift ;;
            -R|--no-resume) RESUME=false; shift ;;
            -C|--cleanup) CLEANUP=true; shift ;;
            --no-selfies) CONVERT_TO_SELFIES=false; shift ;;
            --selfies-output) OUTPUT_SELFIES="$2"; shift 2 ;;
            -s|--status) show_status; exit 0 ;;
            -k|--kill) kill_pipeline; exit 0 ;;
            --debug) DEBUG_MODE=true; shift ;;
            --usearch) DOWNLOAD_METHOD="usearch"; shift ;;
            --ftp) DOWNLOAD_METHOD="ftp"; shift ;;
            -h|--help) show_usage; exit 0 ;;
            *) log_error "Unknown option: $1"; show_usage; exit 1 ;;
        esac
    done
    
    mkdir -p "$LOG_DIR" "$DATA_DIR/cache" "$DATA_DIR/output"
    
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