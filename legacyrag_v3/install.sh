#!/usr/bin/env bash
# install.sh — PhaseRAG v3 Zero-Config Installer
# Detects hardware, picks model, configures llama-server, starts system.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$REPO_DIR/legacyrag_config.json"
LOG_FILE="$REPO_DIR/results/install.log"
mkdir -p "$REPO_DIR/results"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
die() { log "ERROR: $*"; exit 1; }

log "=== PhaseRAG v3 Installer ==="
log "Detecting hardware..."

# ── OS check ─────────────────────────────────────────────────────────────────
OS_ID=$(. /etc/os-release && echo "$ID")
OS_VERSION=$(. /etc/os-release && echo "$VERSION_ID")
log "OS: $OS_ID $OS_VERSION"
[[ "$OS_ID" == "ubuntu" ]] || log "WARNING: Not Ubuntu — installer tested on Ubuntu 20.04/22.04/24.04 only"

# ── CPU detection ─────────────────────────────────────────────────────────────
PHYSICAL_CORES=$(lscpu | grep "^Core(s) per socket" | awk '{print $NF}')
SOCKETS=$(lscpu | grep "^Socket(s)" | awk '{print $NF}')
TOTAL_CORES=$((PHYSICAL_CORES * SOCKETS))
CPU_MODEL=$(lscpu | grep "Model name" | sed 's/.*: *//')
THREADS=$((TOTAL_CORES < 8 ? TOTAL_CORES : 8))
log "CPU: $CPU_MODEL ($TOTAL_CORES physical cores)"

# ── RAM detection ─────────────────────────────────────────────────────────────
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
log "RAM: ${RAM_GB} GB"

# ── GPU detection ─────────────────────────────────────────────────────────────
GPU_COUNT=0
TOTAL_VRAM_MB=0
NGL=0

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1}END{print s}')
    log "GPU count: $GPU_COUNT, Total VRAM: ${TOTAL_VRAM_MB} MB"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r idx name vram; do
        log "  GPU $idx: $name — ${vram// /} MB"
    done
    NGL=99
else
    log "No NVIDIA GPU detected — CPU-only mode"
fi

# ── Model selection ────────────────────────────────────────────────────────────
USABLE_VRAM=$((TOTAL_VRAM_MB - 1024))  # reserve 1GB
if [[ $GPU_COUNT -eq 0 ]]; then
    MODEL="phi3:mini"; QUANT="Q4_K_M"; NGL=0
    log "Model: $MODEL $QUANT (CPU-only)"
elif [[ $USABLE_VRAM -lt 2000 ]]; then
    MODEL="qwen2:0.5b"; QUANT="Q4_K_M"
    log "Model: $MODEL $QUANT (<2GB VRAM)"
elif [[ $USABLE_VRAM -lt 4000 ]]; then
    MODEL="phi3:mini"; QUANT="Q4_K_M"
    log "Model: $MODEL $QUANT (2-4GB VRAM)"
elif [[ $USABLE_VRAM -lt 8000 ]]; then
    MODEL="phi3:mini"; QUANT="Q4_K_M"
    log "Model: $MODEL $QUANT (4-8GB VRAM, dual K4200 detected)"
elif [[ $USABLE_VRAM -lt 16000 ]]; then
    MODEL="qwen2.5:7b-instruct-q2_K"; QUANT="Q2_K"
    log "Model: $MODEL $QUANT (8-16GB VRAM)"
else
    MODEL="qwen2.5:7b-instruct-q4_K_M"; QUANT="Q4_K_M"
    log "Model: $MODEL $QUANT (>16GB VRAM)"
fi

# ── Check Ollama ──────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    log "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
ollama serve &>/dev/null &
sleep 3

# ── Pull model if not present ─────────────────────────────────────────────────
if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    log "Pulling model: $MODEL..."
    ollama pull "$MODEL"
else
    log "Model $MODEL already present."
fi

# ── Find llama-server binary ──────────────────────────────────────────────────
LLAMA_BIN=""
for candidate in \
    "$REPO_DIR/../build_b9297/llama-server" \
    "$REPO_DIR/../build/bin/llama-server" \
    "$(which llama-server 2>/dev/null || true)"; do
    if [[ -x "$candidate" ]]; then
        LLAMA_BIN="$candidate"
        break
    fi
done

if [[ -z "$LLAMA_BIN" ]]; then
    log "llama-server not found. Please build llama.cpp with Vulkan support."
    log "See: https://github.com/ggml-org/llama.cpp#build"
    die "llama-server not found"
fi
log "llama-server: $LLAMA_BIN"

# ── Find model GGUF ───────────────────────────────────────────────────────────
MODEL_SLUG="${MODEL//:/_}"
MANIFEST_BASE="${MODEL%%:*}"
MANIFEST_TAG="${MODEL##*:}"
MANIFEST="/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/$MANIFEST_BASE/$MANIFEST_TAG"

if [[ ! -f "$MANIFEST" ]]; then
    die "Manifest not found at $MANIFEST — did model pull succeed?"
fi
DIGEST=$(python3 -c "
import json
with open('$MANIFEST') as f: d=json.load(f)
for l in d['layers']:
    if l['mediaType']=='application/vnd.ollama.image.model':
        print(l['digest'].replace('sha256:','sha256-')); break
")
MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/$DIGEST"
[[ -f "$MODEL_PATH" ]] || die "Model GGUF not found at $MODEL_PATH"
log "Model GGUF: $MODEL_PATH"

# ── Write config ──────────────────────────────────────────────────────────────
cat > "$CONFIG_FILE" <<EOF
{
  "model": "$MODEL",
  "quantization": "$QUANT",
  "model_path": "$MODEL_PATH",
  "llama_server": "$LLAMA_BIN",
  "ngl": $NGL,
  "threads": $THREADS,
  "ctx_size": 2048,
  "parallel": 1,
  "port": 8080,
  "ngram_speculative": $([ $NGL -eq 99 ] && echo "true" || echo "false"),
  "hardware": {
    "cpu": "$CPU_MODEL",
    "physical_cores": $TOTAL_CORES,
    "ram_gb": $RAM_GB,
    "gpu_count": $GPU_COUNT,
    "total_vram_mb": $TOTAL_VRAM_MB
  }
}
EOF
log "Config written to $CONFIG_FILE"

# ── Start llama-server ────────────────────────────────────────────────────────
log "Starting llama-server..."
BIN_DIR="$(dirname "$LLAMA_BIN")"
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

SERVER_ARGS=(
    "-m" "$MODEL_PATH"
    "-ngl" "$NGL"
    "--port" "8080"
    "--host" "0.0.0.0"
    "--ctx-size" "2048"
    "--threads" "$THREADS"
    "--parallel" "1"
)
if [[ $NGL -eq 99 ]]; then
    SERVER_ARGS+=("--spec-type" "ngram-simple" "--spec-draft-n-max" "8")
fi

nohup "$LLAMA_BIN" "${SERVER_ARGS[@]}" > "$REPO_DIR/results/server.log" 2>&1 &
SERVER_PID=$!
log "llama-server PID: $SERVER_PID"

# Wait for health
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
        log "llama-server is healthy"
        break
    fi
    sleep 2
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        die "llama-server exited. Check $REPO_DIR/results/server.log"
    fi
done

# ── Start Web UI ──────────────────────────────────────────────────────────────
log "Starting Web UI on port 7860..."
cd "$REPO_DIR"
nohup python3 -m uvicorn web_ui.app:app --host 0.0.0.0 --port 7860 \
    > "$REPO_DIR/results/webui.log" 2>&1 &
WEBUI_PID=$!
sleep 3

log ""
log "=========================================="
log "PhaseRAG v3 is running!"
log "  Web UI:       http://$(hostname -I | awk '{print $1}'):7860"
log "  API:          http://$(hostname -I | awk '{print $1}'):8080"
log "  Model:        $MODEL ($QUANT)"
log "  GPU layers:   $NGL"
log "  Threads:      $THREADS"
log ""
log "Expected throughput: ~8-9 tok/s (dual K4200 + ngram)"
log "  Short query (~50 tokens):   ~25s"
log "  Medium query (~200 tokens): ~150s"
log "  Long prompt (~400 tokens):  ~400s"
log "=========================================="
