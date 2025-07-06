#!/bin/bash

# =============================================================================
# PubChem SMILES Download Script
# =============================================================================
# Robust bash wrapper for downloading and processing PubChem compound data
# with resumable downloads, progress tracking, and comprehensive error handling
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/prepare_data.py"
LOG_DIR="$SCRIPT_DIR/logs"
STATE_FILE="$LOG_DIR/download_state.json"
PROGRESS_FILE="$LOG_DIR/download_progress.log"
ERROR_LOG="$LOG_DIR/download_errors.log"

# Default values
DEFAULT_LIMIT=10000000
DEFAULT_OUTPUT="training.txt"
DEFAULT_DATA_DIR="data"
DEFAULT_CACHE_SUBDIR="cache"
DEFAULT_OUTPUT_SUBDIR="output"
DEFAULT_MAX_FILES=""
DEFAULT_CLEANUP=false
DEFAULT_RESUME=true
DEFAULT_CONVERT_TO_SELFIES=true
DEFAULT_SELFIES_OUTPUT=""
DEBUG_MODE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking
TOTAL_DOWNLOADED=0
TOTAL_PROCESSED=0
TOTAL_SMILES=0
START_TIME=$(date +%s)

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROGRESS_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROGRESS_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$ERROR_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROGRESS_FILE"
}

show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                      PubChem SMILES/SELFIES Downloader                      ║"
    echo "║                        Modern ML-Ready Chemical Datasets                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${BLUE}Features: Multi-threaded downloads • Progress tracking • SELFIES conversion${NC}"
    echo -e "${BLUE}Default: $(printf "%'d" $DEFAULT_LIMIT) molecules → training.txt + training_selfies.txt${NC}"
    echo ""
}

show_usage() {
    echo -e "${CYAN}PubChem SMILES/SELFIES Downloader${NC}"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo -e "${GREEN}Basic Options:${NC}"
    echo "  -l, --limit LIMIT        Number of SMILES to collect (default: $(printf "%'d" $DEFAULT_LIMIT))"
    echo "  -o, --output OUTPUT      Output filename (default: $DEFAULT_OUTPUT)"
    echo "  -d, --data-dir DIR       Data directory (default: $DEFAULT_DATA_DIR)"
    echo ""
    echo -e "${GREEN}Advanced Options:${NC}"
    echo "  -c, --cache-subdir DIR   Cache subdirectory (default: $DEFAULT_CACHE_SUBDIR)"
    echo "  -u, --output-subdir DIR  Output subdirectory (default: $DEFAULT_OUTPUT_SUBDIR)"
    echo "  -m, --max-files NUM      Maximum files to process (default: all available)"
    echo ""
    echo -e "${GREEN}Conversion Options:${NC}"
    echo "  -N, --no-convert         Disable SELFIES conversion (enabled by default)"
    echo "  -f, --selfies-output     SELFIES output filename (auto-generated if not specified)"
    echo ""
    echo -e "${GREEN}Process Control:${NC}"
    echo "  -r, --resume             Resume previous download (default: enabled)"
    echo "  -R, --no-resume          Start fresh download, ignore previous state"
    echo "  -C, --cleanup            Clean up cache after completion"
    echo "  -s, --status             Show current download status"
    echo "  -k, --kill               Kill running download process"
    echo "  --debug              Enable debug mode (more verbose logging)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                           # Download 10M molecules, convert to SELFIES"
    echo "  $0 -l 5000000               # Download 5M molecules with SELFIES"
    echo "  $0 -N                       # Download SMILES only (no SELFIES conversion)"
    echo "  $0 -m 5 -l 100000           # Test run: 5 files, 100K molecules"
    echo "  $0 -f my_dataset.txt        # Custom SELFIES filename"
    echo "  $0 -s                       # Check current download status"
    echo "  $0 -k                       # Stop running download"
    echo ""
    echo -e "${BLUE}Note: SELFIES conversion is enabled by default for modern ML workflows${NC}"
}

# =============================================================================
# System Checks
# =============================================================================

check_dependencies() {
    log_info "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check aria2c
    if ! command -v aria2c &> /dev/null; then
        missing_deps+=("aria2c")
    fi
    
    # Check required Python packages
    if ! python3 -c "import requests" &> /dev/null; then
        missing_deps+=("python3-requests")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        echo -e "${RED}Please install missing dependencies:${NC}"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

check_disk_space() {
    local cache_dir="$1"
    local required_gb=50  # Minimum 50GB recommended
    
    log_info "Checking available disk space..."
    
    local available_kb=$(df "$cache_dir" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [ $available_gb -lt $required_gb ]; then
        log_warn "Low disk space: ${available_gb}GB available, ${required_gb}GB recommended"
        echo -e "${YELLOW}Continue anyway? (y/N)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Download cancelled by user"
            exit 0
        fi
    else
        log_success "Sufficient disk space: ${available_gb}GB available"
    fi
}

# =============================================================================
# State Management
# =============================================================================

save_state() {
    local limit="$1"
    local output="$2"
    local data_dir="$3"
    local cache_subdir="$4"
    local output_subdir="$5"
    local max_files="$6"
    local convert_to_selfies="$7"
    local selfies_output="$8"
    local pid="$9"
    
    mkdir -p "$LOG_DIR"
    
    cat > "$STATE_FILE" << EOF
{
    "limit": $limit,
    "output": "$output",
    "data_dir": "$data_dir",
    "cache_subdir": "$cache_subdir",
    "output_subdir": "$output_subdir",
    "max_files": "$max_files",
    "convert_to_selfies": $convert_to_selfies,
    "selfies_output": "$selfies_output",
    "pid": $pid,
    "start_time": $START_TIME,
    "status": "running"
}
EOF
}

load_state() {
    if [ -f "$STATE_FILE" ]; then
        if command -v jq &> /dev/null; then
            # Use jq if available
            echo "$(cat "$STATE_FILE")"
        else
            # Fallback to simple parsing
            cat "$STATE_FILE"
        fi
    else
        echo "{}"
    fi
}

update_state() {
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
# Progress Monitoring
# =============================================================================

show_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "\r${CYAN}Progress: [${NC}"
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    printf "${CYAN}] %d%% (%d/%d)${NC}" $percentage $current $total
}

monitor_python_process() {
    local pid=$1
    local limit=$2
    
    log_info "Monitoring download process (PID: $pid)"
    log_info "Progress updates every 3 seconds..."
    
    local last_collected=0
    local stall_count=0
    
    while kill -0 $pid 2>/dev/null; do
        # Parse progress from Python output
        if [ -f "$PROGRESS_FILE" ]; then
            local collected=$(tail -n 100 "$PROGRESS_FILE" | grep -o "Total collected: [0-9,]*" | tail -1 | grep -o "[0-9,]*" | tr -d ',')
            local current_file=$(tail -n 20 "$PROGRESS_FILE" | grep -o "Processing file [0-9]*/[0-9]*" | tail -1)
            local download_status=$(tail -n 10 "$PROGRESS_FILE" | grep -E "(Downloading|Successfully|Failed)" | tail -1)
            
            if [ -n "$collected" ] && [ "$collected" -gt 0 ]; then
                show_enhanced_progress "$collected" "$limit" "$current_file" "$download_status"
                TOTAL_SMILES=$collected
                
                # Check for stalled progress
                if [ "$collected" -eq "$last_collected" ]; then
                    ((stall_count++))
                    if [ $stall_count -ge 10 ]; then
                        log_warn "Progress appears stalled. Process may be downloading large file..."
                        stall_count=0
                    fi
                else
                    stall_count=0
                    last_collected=$collected
                fi
            fi
        fi
        
        sleep 3
    done
    
    echo  # New line after progress bar
    log_info "Process monitoring completed"
}

show_enhanced_progress() {
    local current=$1
    local total=$2
    local file_info="$3"
    local download_status="$4"
    
    local percentage=$((current * 100 / total))
    local width=40
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    # Create gradient progress bar
    local bar=""
    for ((i=0; i<filled; i++)); do
        bar+="█"
    done
    for ((i=0; i<empty; i++)); do
        bar+="░"
    done
    
    # Calculate ETA
    local elapsed=$(($(date +%s) - START_TIME))
    local rate=$((current * 3600 / elapsed))
    local remaining=$((total - current))
    local eta_hours=$((remaining / rate))
    local eta_str="∞"
    
    if [ $rate -gt 0 ] && [ $eta_hours -lt 24 ]; then
        if [ $eta_hours -gt 0 ]; then
            eta_str="${eta_hours}h"
        else
            local eta_minutes=$((remaining * 60 / rate))
            eta_str="${eta_minutes}m"
        fi
    fi
    
    printf "\rProgress: [%s] %d%% (%'d/%'d) | Rate: %'d/h | ETA: %s" \
           "$bar" "$percentage" "$current" "$total" "$rate" "$eta_str"
    
    # Show file info on next line if available
    if [ -n "$file_info" ]; then
        printf "\n%s" "$file_info"
        printf "\r"
    fi
}

# =============================================================================
# Download Management
# =============================================================================

start_download() {
    local limit="$1"
    local output="$2"
    local data_dir="$3"
    local cache_subdir="$4"
    local output_subdir="$5"
    local max_files="$6"
    local cleanup="$7"
    local convert_to_selfies="$8"
    local selfies_output="$9"
    
    # Set start time for progress monitoring
    START_TIME=$(date +%s)
    
    if [ "$DEBUG_MODE" = true ]; then
        log_info "Debug mode enabled. Clearing logs."
        rm -f "$LOG_DIR"/*
    fi

    echo -e "${GREEN}Starting PubChem Download Pipeline${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log_info "Target molecules: $(printf "%'d" "$limit")"
    log_info "SMILES output: $output"
    
    if [ "$convert_to_selfies" = true ]; then
        local selfies_name="$selfies_output"
        if [ -z "$selfies_name" ]; then
            local stem=$(basename "$output" .txt)
            selfies_name="${stem}_selfies.txt"
        fi
        log_info "SELFIES output: $selfies_name"
        log_info "Conversion: ENABLED (modern ML format)"
    else
        log_info "SELFIES conversion: DISABLED"
    fi
    
    log_info "Data directory: $data_dir"
    log_info "Cache location: $data_dir/$cache_subdir"
    log_info "Output location: $data_dir/$output_subdir"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check disk space
    check_disk_space "$data_dir/$cache_subdir"
    
    # Prepare Python command
    local python_cmd="python3 '$PYTHON_SCRIPT' --limit $limit --output '$output' --data-dir '$data_dir' --cache-subdir '$cache_subdir' --output-subdir '$output_subdir'"
    
    if [ "$DEBUG_MODE" = true ]; then
        python_cmd="$python_cmd --debug"
    fi

    if [ -n "$max_files" ]; then
        python_cmd="$python_cmd --max-files $max_files"
    fi
    
    if [ "$cleanup" = true ]; then
        python_cmd="$python_cmd --cleanup"
    fi
    
    if [ "$convert_to_selfies" = false ]; then
        python_cmd="$python_cmd --no-convert-to-selfies"
    else
        if [ -n "$selfies_output" ]; then
            python_cmd="$python_cmd --selfies-output '$selfies_output'"
        fi
    fi
    
    # Redirect Python output to log file
    python_cmd="$python_cmd 2>&1 | tee -a '$PROGRESS_FILE'"
    
    # Start Python process in background
    eval "$python_cmd" &
    local pid=$!
    
    # Save state
    save_state "$limit" "$output" "$data_dir" "$cache_subdir" "$output_subdir" "$max_files" "$convert_to_selfies" "$selfies_output" "$pid"
    
    # Monitor progress
    monitor_python_process "$pid" "$limit"
    
    # Wait for completion
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Download completed successfully!"
        update_state "completed"
        show_completion_stats
    else
        log_error "Download failed with exit code: $exit_code"
        update_state "failed"
        return $exit_code
    fi
}

show_completion_stats() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    # Check if SELFIES was generated
    local selfies_info=""
    if [ -f "$data_dir/$output_subdir/${output%.*}_selfies.txt" ]; then
        selfies_info="SELFIES dataset ready for ML training"
    fi
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                             DOWNLOAD COMPLETED                              ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║ Molecules Collected: $(printf "%'12d" "$TOTAL_SMILES")                                      ║"
    echo "║ Total Duration:      $(printf "%02d:%02d:%02d" $hours $minutes $seconds)                                           ║"
    echo "║ Average Rate:        $(printf "%'12.0f" $((TOTAL_SMILES * 3600 / duration))) molecules/hour                         ║"
    if [ -n "$selfies_info" ]; then
        echo "║ $selfies_info                                 ║"
    fi
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║ Output Files:                                                               ║"
    echo "║   • SMILES: $output                                              ║"
    if [ -n "$selfies_info" ]; then
        echo "║   • SELFIES: ${output%.*}_selfies.txt                                  ║"
    fi
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "   • Check output files in: $data_dir/$output_subdir/"
    echo -e "   • Use --cleanup to free disk space"
    echo -e "   • Ready for ML training and analysis!"
}

# =============================================================================
# Status and Control
# =============================================================================

show_status() {
    if [ ! -f "$STATE_FILE" ]; then
        echo -e "${YELLOW}No download state found${NC}"
        return 0
    fi
    
    local state=$(load_state)
    
    echo -e "${CYAN}Current Download Status:${NC}"
    echo "========================="
    
    if command -v jq &> /dev/null; then
        local status=$(echo "$state" | jq -r '.status // "unknown"')
        local limit=$(echo "$state" | jq -r '.limit // "unknown"')
        local pid=$(echo "$state" | jq -r '.pid // "unknown"')
        local start_time=$(echo "$state" | jq -r '.start_time // "unknown"')
        
        echo "Status: $status"
        echo "Target SMILES: $(printf "%'d" "$limit")"
        echo "Process ID: $pid"
        
        if [ "$pid" != "unknown" ] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}Process is running${NC}"
        else
            echo -e "${RED}Process is not running${NC}"
        fi
        
        if [ "$start_time" != "unknown" ]; then
            local current_time=$(date +%s)
            local duration=$((current_time - start_time))
            local hours=$((duration / 3600))
            local minutes=$(((duration % 3600) / 60))
            echo "Running time: ${hours}h ${minutes}m"
        fi
    else
        echo "State file exists but jq not available for detailed parsing"
        cat "$STATE_FILE"
    fi
    
    # Show recent progress
    if [ -f "$PROGRESS_FILE" ]; then
        echo -e "\n${CYAN}Recent Progress:${NC}"
        echo "================"
        tail -n 10 "$PROGRESS_FILE"
    fi
}

kill_download() {
    local state=$(load_state)
    
    if command -v jq &> /dev/null; then
        local pid=$(echo "$state" | jq -r '.pid // ""')
        
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log_info "Killing download process (PID: $pid)"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 3
            
            if kill -0 "$pid" 2>/dev/null; then
                log_warn "Process still running, force killing..."
                kill -KILL "$pid" 2>/dev/null || true
            fi
            
            update_state "killed"
            log_success "Download process terminated"
        else
            log_warn "No running download process found"
        fi
    else
        log_error "Cannot kill process - jq not available"
    fi
}

# =============================================================================
# Signal Handlers
# =============================================================================

cleanup_on_exit() {
    if [ -n "${DOWNLOAD_PID:-}" ]; then
        log_info "Cleaning up on exit..."
        kill -TERM "$DOWNLOAD_PID" 2>/dev/null || true
        update_state "interrupted"
    fi
}

trap cleanup_on_exit EXIT INT TERM

# =============================================================================
# Main Function
# =============================================================================

main() {
    local limit="$DEFAULT_LIMIT"
    local output="$DEFAULT_OUTPUT"
    local data_dir="$DEFAULT_DATA_DIR"
    local cache_subdir="$DEFAULT_CACHE_SUBDIR"
    local output_subdir="$DEFAULT_OUTPUT_SUBDIR"
    local max_files="$DEFAULT_MAX_FILES"
    local cleanup="$DEFAULT_CLEANUP"
    local resume="$DEFAULT_RESUME"
    local convert_to_selfies="$DEFAULT_CONVERT_TO_SELFIES"
    local selfies_output="$DEFAULT_SELFIES_OUTPUT"
    local show_status_only=false
    local kill_process=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -l|--limit)
                limit="$2"
                shift 2
                ;;
            -o|--output)
                output="$2"
                shift 2
                ;;
            -d|--data-dir)
                data_dir="$2"
                shift 2
                ;;
            -c|--cache-subdir)
                cache_subdir="$2"
                shift 2
                ;;
            -u|--output-subdir)
                output_subdir="$2"
                shift 2
                ;;
            -m|--max-files)
                max_files="$2"
                shift 2
                ;;
            -r|--resume)
                resume=true
                shift
                ;;
            -R|--no-resume)
                resume=false
                shift
                ;;
            -C|--cleanup)
                cleanup=true
                shift
                ;;
            -N|--no-convert)
                convert_to_selfies=false
                shift
                ;;
            -f|--selfies-output)
                selfies_output="$2"
                shift 2
                ;;
            -s|--status)
                show_status_only=true
                shift
                ;;
            -k|--kill)
                kill_process=true
                shift
                ;;
            --debug)
                DEBUG_MODE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Create necessary directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$data_dir/$cache_subdir"
    mkdir -p "$data_dir/$output_subdir"
    
    # Handle special commands
    if [ "$show_status_only" = true ]; then
        show_status
        exit 0
    fi
    
    if [ "$kill_process" = true ]; then
        kill_download
        exit 0
    fi
    
    # Show banner
    show_banner
    
    # Check dependencies
    check_dependencies
    
    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Handle resume logic
    if [ "$resume" = true ] && [ -f "$STATE_FILE" ]; then
        local state=$(load_state)
        if command -v jq &> /dev/null; then
            local status=$(echo "$state" | jq -r '.status // "unknown"')
            local pid=$(echo "$state" | jq -r '.pid // ""')
            
            if [ "$status" = "running" ] && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                log_info "Download already running (PID: $pid)"
                echo -e "${YELLOW}Continue monitoring? (y/N)${NC}"
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    monitor_python_process "$pid" "$limit"
                    exit 0
                fi
            fi
        fi
    fi
    
    # Start download
    start_download "$limit" "$output" "$data_dir" "$cache_subdir" "$output_subdir" "$max_files" "$cleanup" "$convert_to_selfies" "$selfies_output"
}

# =============================================================================
# Script Entry Point
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 