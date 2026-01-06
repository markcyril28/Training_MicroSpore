#!/bin/bash
#===============================================================================
# Microspore Phenotyping - Logging Utilities
# Comprehensive logging for ML training: GPU, system, errors, metrics
#
# Location: modules/logging/logging_utils.sh
# Source:   source "${BASE_DIR}/modules/logging/logging_utils.sh"
#===============================================================================

# Get the script directory for relative paths
LOGGING_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate up 2 levels: logging -> modules -> BASE_DIR
LOGGING_BASE_DIR="$(dirname "$(dirname "$LOGGING_SCRIPT_DIR")")"

#===============================================================================
# LOGGING CONFIGURATION
#===============================================================================

# Base logs directory
LOGS_BASE_DIR="${LOGGING_BASE_DIR}/logs"

# Log subdirectories
FULL_LOGS_DIR="${LOGS_BASE_DIR}/full_logs"
GPU_LOGS_DIR="${LOGS_BASE_DIR}/gpu_logs"
SYSTEM_LOGS_DIR="${LOGS_BASE_DIR}/system_resources_log"
ERROR_LOGS_DIR="${LOGS_BASE_DIR}/errors_logs"
VIS_LOGS_DIR="${LOGS_BASE_DIR}/visualization_logs"
METRICS_LOGS_DIR="${LOGS_BASE_DIR}/training_metrics"

# GPU monitoring interval (seconds)
GPU_MONITOR_INTERVAL=5

# System monitoring interval (seconds)
SYSTEM_MONITOR_INTERVAL=10

# PID files for background monitors
GPU_MONITOR_PID_FILE="/tmp/gpu_monitor_$$.pid"
SYSTEM_MONITOR_PID_FILE="/tmp/system_monitor_$$.pid"

#===============================================================================
# DIRECTORY SETUP
#===============================================================================

# Initialize all logging directories
# IMPORTANT: Call this function directly (not in subshell) to set variables properly
# Usage: init_logging_dirs "experiment_name"
#        Then use $EXPERIMENT_LOG_DIR
init_logging_dirs() {
    local experiment_name="${1:-default}"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Create experiment-specific subdirectories
    EXPERIMENT_LOG_DIR="${LOGS_BASE_DIR}/${experiment_name}_${timestamp}"
    
    mkdir -p "${EXPERIMENT_LOG_DIR}/full_logs"
    mkdir -p "${EXPERIMENT_LOG_DIR}/gpu_logs"
    mkdir -p "${EXPERIMENT_LOG_DIR}/system_resources_log"
    mkdir -p "${EXPERIMENT_LOG_DIR}/errors_logs"
    mkdir -p "${EXPERIMENT_LOG_DIR}/visualization_logs"
    mkdir -p "${EXPERIMENT_LOG_DIR}/training_metrics"
    
    # Set current log paths as GLOBAL variables (not local)
    CURRENT_FULL_LOG="${EXPERIMENT_LOG_DIR}/full_logs/training_${timestamp}.log"
    CURRENT_GPU_LOG="${EXPERIMENT_LOG_DIR}/gpu_logs/gpu_${timestamp}.csv"
    CURRENT_SYSTEM_LOG="${EXPERIMENT_LOG_DIR}/system_resources_log/system_${timestamp}.csv"
    CURRENT_ERROR_LOG="${EXPERIMENT_LOG_DIR}/errors_logs/errors_${timestamp}.log"
    CURRENT_METRICS_DIR="${EXPERIMENT_LOG_DIR}/training_metrics"
    CURRENT_VIS_DIR="${EXPERIMENT_LOG_DIR}/visualization_logs"
    
    # Mark logging as initialized
    LOGGING_INITIALIZED=true
    
    # Initialize CSV headers
    echo "timestamp,gpu_id,gpu_name,gpu_util_pct,memory_used_mb,memory_total_mb,memory_pct,temperature_c,power_w" > "$CURRENT_GPU_LOG"
    echo "timestamp,cpu_pct,ram_used_mb,ram_total_mb,ram_pct,disk_read_mb_s,disk_write_mb_s" > "$CURRENT_SYSTEM_LOG"
    
    # Initialize error log
    touch "$CURRENT_ERROR_LOG"
    
    # Log initialization
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Logging initialized for experiment: ${experiment_name}" >> "$CURRENT_FULL_LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log directory: ${EXPERIMENT_LOG_DIR}" >> "$CURRENT_FULL_LOG"
    
    # Print confirmation (not to be captured)
    echo "[Logging] Initialized: ${EXPERIMENT_LOG_DIR}" >&2
    
    return 0
}

#===============================================================================
# FULL LOGGING (tee wrapper)
#===============================================================================

# Check if logging is initialized
_check_logging_init() {
    if [ "$LOGGING_INITIALIZED" != "true" ] || [ -z "$CURRENT_FULL_LOG" ]; then
        echo "[Logging] WARNING: Logging not initialized. Call init_logging_dirs first." >&2
        return 1
    fi
    return 0
}

# Log message to full log and stdout
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if _check_logging_init; then
        echo "[${timestamp}] ${message}" | tee -a "$CURRENT_FULL_LOG"
    else
        echo "[${timestamp}] ${message}"
    fi
}

# Log info
log_info() {
    log_message "[INFO] $1"
}

# Log warning
log_warning() {
    local message="$1"
    log_message "[WARNING] $message"
    if _check_logging_init; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $message" >> "$CURRENT_ERROR_LOG"
    fi
}

# Log error
log_error() {
    local message="$1"
    log_message "[ERROR] $message"
    if _check_logging_init; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $message" >> "$CURRENT_ERROR_LOG"
    fi
}

# Start full output logging (captures all stdout/stderr)
start_full_logging() {
    if ! _check_logging_init; then
        return 1
    fi
    exec > >(tee -a "$CURRENT_FULL_LOG") 2>&1
    log_info "Full logging started"
}

#===============================================================================
# GPU MONITORING
#===============================================================================

# Get GPU stats as CSV line
get_gpu_stats() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null | while read line; do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            # Use sed to trim whitespace (safer than xargs for special characters)
            local gpu_id=$(echo "$line" | cut -d',' -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            local gpu_name=$(echo "$line" | cut -d',' -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            local gpu_util=$(echo "$line" | cut -d',' -f3 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            local mem_used=$(echo "$line" | cut -d',' -f4 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            local mem_total=$(echo "$line" | cut -d',' -f5 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            local temp=$(echo "$line" | cut -d',' -f6 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            local power=$(echo "$line" | cut -d',' -f7 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            # Calculate memory percentage
            local mem_pct=0
            if [ "$mem_total" != "0" ] && [ -n "$mem_total" ]; then
                mem_pct=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "0")
            fi
            
            echo "${timestamp},${gpu_id},${gpu_name},${gpu_util},${mem_used},${mem_total},${mem_pct},${temp},${power}"
        done
    fi
}

# Background GPU monitor
_gpu_monitor_loop() {
    local log_file="$1"
    local interval="$2"
    
    while true; do
        get_gpu_stats >> "$log_file"
        sleep "$interval"
    done
}

# Start GPU monitoring in background
start_gpu_monitor() {
    local interval="${1:-$GPU_MONITOR_INTERVAL}"
    
    if ! _check_logging_init; then
        return 1
    fi
    
    _gpu_monitor_loop "$CURRENT_GPU_LOG" "$interval" &
    local pid=$!
    echo $pid > "$GPU_MONITOR_PID_FILE"
    
    log_info "GPU monitoring started (PID: $pid, interval: ${interval}s)"
    echo $pid
}

# Stop GPU monitoring
stop_gpu_monitor() {
    if [ -f "$GPU_MONITOR_PID_FILE" ]; then
        local pid=$(cat "$GPU_MONITOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            log_info "GPU monitoring stopped (PID: $pid)"
        fi
        rm -f "$GPU_MONITOR_PID_FILE"
    fi
}

#===============================================================================
# SYSTEM RESOURCE MONITORING
#===============================================================================

# Get system stats as CSV line
get_system_stats() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # CPU usage
    local cpu_pct=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 2>/dev/null || echo "0")
    
    # RAM usage
    local ram_info=$(free -m | grep Mem)
    local ram_total=$(echo "$ram_info" | awk '{print $2}')
    local ram_used=$(echo "$ram_info" | awk '{print $3}')
    local ram_pct=0
    if [ "$ram_total" != "0" ] && [ -n "$ram_total" ]; then
        ram_pct=$(echo "scale=1; $ram_used * 100 / $ram_total" | bc 2>/dev/null || echo "0")
    fi
    
    # Disk I/O (simplified - requires iostat for detailed)
    local disk_read="0"
    local disk_write="0"
    if command -v iostat &> /dev/null; then
        local iostat_out=$(iostat -d 1 2 | tail -n +4 | head -1)
        disk_read=$(echo "$iostat_out" | awk '{print $3}' 2>/dev/null || echo "0")
        disk_write=$(echo "$iostat_out" | awk '{print $4}' 2>/dev/null || echo "0")
    fi
    
    echo "${timestamp},${cpu_pct},${ram_used},${ram_total},${ram_pct},${disk_read},${disk_write}"
}

# Background system monitor
_system_monitor_loop() {
    local log_file="$1"
    local interval="$2"
    
    while true; do
        get_system_stats >> "$log_file"
        sleep "$interval"
    done
}

# Start system monitoring in background
start_system_monitor() {
    local interval="${1:-$SYSTEM_MONITOR_INTERVAL}"
    
    if ! _check_logging_init; then
        return 1
    fi
    
    _system_monitor_loop "$CURRENT_SYSTEM_LOG" "$interval" &
    local pid=$!
    echo $pid > "$SYSTEM_MONITOR_PID_FILE"
    
    log_info "System monitoring started (PID: $pid, interval: ${interval}s)"
    echo $pid
}

# Stop system monitoring
stop_system_monitor() {
    if [ -f "$SYSTEM_MONITOR_PID_FILE" ]; then
        local pid=$(cat "$SYSTEM_MONITOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            log_info "System monitoring stopped (PID: $pid)"
        fi
        rm -f "$SYSTEM_MONITOR_PID_FILE"
    fi
}

#===============================================================================
# ERROR LOGGING
#===============================================================================

# Log OOM event
log_oom_event() {
    local details="${1:-Out of memory event detected}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[${timestamp}] [OOM] ${details}" >> "$CURRENT_ERROR_LOG"
    
    # Also capture GPU memory state
    if command -v nvidia-smi &> /dev/null; then
        echo "[${timestamp}] [OOM] GPU Memory State:" >> "$CURRENT_ERROR_LOG"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$CURRENT_ERROR_LOG"
    fi
    
    log_error "OOM Event: ${details}"
}

# Log training interruption
log_interruption() {
    local reason="${1:-Training interrupted}"
    local recovery_point="${2:-unknown}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[${timestamp}] [INTERRUPTION] Reason: ${reason}" >> "$CURRENT_ERROR_LOG"
    echo "[${timestamp}] [INTERRUPTION] Recovery point: ${recovery_point}" >> "$CURRENT_ERROR_LOG"
    
    log_warning "Training interrupted: ${reason} (Recovery: ${recovery_point})"
}

# Log warning aggregation
log_training_warning() {
    local warning_type="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[${timestamp}] [${warning_type}] ${message}" >> "$CURRENT_ERROR_LOG"
}

#===============================================================================
# METRICS LOGGING (calls Python module)
#===============================================================================

# Save training metrics snapshot
save_metrics_snapshot() {
    local epoch="$1"
    local train_loss="$2"
    local val_loss="$3"
    local map50="$4"
    local map50_95="$5"
    local lr="$6"
    
    local metrics_file="${CURRENT_METRICS_DIR}/metrics.csv"
    
    # Create header if file doesn't exist
    if [ ! -f "$metrics_file" ]; then
        echo "timestamp,epoch,train_loss,val_loss,mAP50,mAP50-95,learning_rate" > "$metrics_file"
    fi
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${timestamp},${epoch},${train_loss},${val_loss},${map50},${map50_95},${lr}" >> "$metrics_file"
}

# Log best model checkpoint
log_best_checkpoint() {
    local epoch="$1"
    local metric_name="$2"
    local metric_value="$3"
    local checkpoint_path="$4"
    
    local checkpoint_log="${CURRENT_METRICS_DIR}/best_checkpoints.log"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[${timestamp}] Best ${metric_name}: ${metric_value} at epoch ${epoch}" >> "$checkpoint_log"
    echo "  Checkpoint: ${checkpoint_path}" >> "$checkpoint_log"
    
    log_info "New best ${metric_name}: ${metric_value} at epoch ${epoch}"
}

#===============================================================================
# VISUALIZATION LOGGING
#===============================================================================

# Save prediction sample info
log_prediction_sample() {
    local epoch="$1"
    local image_path="$2"
    local output_path="$3"
    
    local samples_log="${CURRENT_VIS_DIR}/prediction_samples.log"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[${timestamp}] Epoch ${epoch}: ${image_path} -> ${output_path}" >> "$samples_log"
}

#===============================================================================
# CLEANUP & SUMMARY
#===============================================================================

# Stop all monitors and generate summary
stop_all_monitors() {
    stop_gpu_monitor
    stop_system_monitor
    if [ "$LOGGING_INITIALIZED" = "true" ]; then
        log_info "All monitors stopped"
    fi
}

# Generate training summary
generate_log_summary() {
    if ! _check_logging_init; then
        echo "[Logging] Cannot generate summary - logging not initialized" >&2
        return 1
    fi
    
    local summary_file="${EXPERIMENT_LOG_DIR}/training_summary.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    {
        echo "=============================================="
        echo "  Training Log Summary"
        echo "  Generated: ${timestamp}"
        echo "=============================================="
        echo ""
        
        # GPU stats summary
        if [ -f "$CURRENT_GPU_LOG" ]; then
            echo "GPU Statistics:"
            echo "  Log file: ${CURRENT_GPU_LOG}"
            local gpu_lines=$(wc -l < "$CURRENT_GPU_LOG")
            echo "  Data points: $((gpu_lines - 1))"
            
            # Calculate averages (skip header)
            if [ $gpu_lines -gt 1 ]; then
                local avg_util=$(tail -n +2 "$CURRENT_GPU_LOG" | cut -d',' -f4 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}')
                local max_mem=$(tail -n +2 "$CURRENT_GPU_LOG" | cut -d',' -f5 | sort -n | tail -1)
                local avg_temp=$(tail -n +2 "$CURRENT_GPU_LOG" | cut -d',' -f8 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}')
                echo "  Avg GPU Utilization: ${avg_util}%"
                echo "  Peak VRAM Used: ${max_mem} MB"
                echo "  Avg Temperature: ${avg_temp}°C"
            fi
            echo ""
        fi
        
        # System stats summary
        if [ -f "$CURRENT_SYSTEM_LOG" ]; then
            echo "System Statistics:"
            echo "  Log file: ${CURRENT_SYSTEM_LOG}"
            local sys_lines=$(wc -l < "$CURRENT_SYSTEM_LOG")
            echo "  Data points: $((sys_lines - 1))"
            
            if [ $sys_lines -gt 1 ]; then
                local avg_cpu=$(tail -n +2 "$CURRENT_SYSTEM_LOG" | cut -d',' -f2 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}')
                local max_ram=$(tail -n +2 "$CURRENT_SYSTEM_LOG" | cut -d',' -f3 | sort -n | tail -1)
                echo "  Avg CPU Usage: ${avg_cpu}%"
                echo "  Peak RAM Used: ${max_ram} MB"
            fi
            echo ""
        fi
        
        # Error summary
        if [ -f "$CURRENT_ERROR_LOG" ]; then
            local error_count=$(grep -c "\[ERROR\]" "$CURRENT_ERROR_LOG" 2>/dev/null || echo "0")
            local warning_count=$(grep -c "\[WARNING\]" "$CURRENT_ERROR_LOG" 2>/dev/null || echo "0")
            local oom_count=$(grep -c "\[OOM\]" "$CURRENT_ERROR_LOG" 2>/dev/null || echo "0")
            
            echo "Error Summary:"
            echo "  Errors: ${error_count}"
            echo "  Warnings: ${warning_count}"
            echo "  OOM Events: ${oom_count}"
            echo ""
        fi
        
        # Metrics summary
        local metrics_file="${CURRENT_METRICS_DIR}/metrics.csv"
        if [ -f "$metrics_file" ]; then
            local epochs_logged=$(($(wc -l < "$metrics_file") - 1))
            echo "Training Metrics:"
            echo "  Epochs logged: ${epochs_logged}"
            
            if [ $epochs_logged -gt 0 ]; then
                local best_map=$(tail -n +2 "$metrics_file" | cut -d',' -f5 | sort -n | tail -1)
                echo "  Best mAP50: ${best_map}"
            fi
            echo ""
        fi
        
        echo "Log Directory: ${EXPERIMENT_LOG_DIR}"
        echo "=============================================="
        
    } > "$summary_file"
    
    cat "$summary_file"
    log_info "Summary saved to: ${summary_file}"
}

# Cleanup function for trap
cleanup_logging() {
    if [ "$LOGGING_INITIALIZED" = "true" ]; then
        log_info "Cleaning up logging..."
        stop_all_monitors
        generate_log_summary
    fi
}

# Set trap for cleanup on exit
trap_logging_cleanup() {
    trap cleanup_logging EXIT INT TERM
}

#===============================================================================
# EXPORT FUNCTIONS
#===============================================================================

export -f init_logging_dirs _check_logging_init
export -f log_message log_info log_warning log_error
export -f start_full_logging
export -f start_gpu_monitor stop_gpu_monitor get_gpu_stats
export -f start_system_monitor stop_system_monitor get_system_stats
export -f log_oom_event log_interruption log_training_warning
export -f save_metrics_snapshot log_best_checkpoint
export -f log_prediction_sample
export -f stop_all_monitors generate_log_summary cleanup_logging trap_logging_cleanup
