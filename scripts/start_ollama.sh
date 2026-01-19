#!/bin/bash
# =============================================================================
# Start Ollama for Agentic 10-K RAG
# =============================================================================
#
# This script starts Ollama natively (recommended for macOS) and pulls
# the required embedding model.
#
# Usage:
#   ./scripts/start_ollama.sh          # Start and pull models
#   ./scripts/start_ollama.sh --check  # Just check if Ollama is running
#
# =============================================================================

set -e

EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
LLM_MODEL="${LLM_MODEL:-}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_ollama_installed() {
    if ! command -v ollama &> /dev/null; then
        log_error "Ollama is not installed."
        echo ""
        echo "Install Ollama:"
        echo "  macOS:  brew install ollama"
        echo "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
        echo "  Or download from: https://ollama.com/download"
        exit 1
    fi
}

check_ollama_running() {
    if curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

start_ollama() {
    log_info "Starting Ollama server..."

    # Start Ollama in the background
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!

    # Wait for Ollama to be ready
    local retries=30
    while [ $retries -gt 0 ]; do
        if check_ollama_running; then
            log_info "Ollama is ready (PID: $OLLAMA_PID)"
            return 0
        fi
        sleep 1
        ((retries--))
    done

    log_error "Ollama failed to start within 30 seconds"
    exit 1
}

pull_model() {
    local model=$1
    log_info "Pulling model: $model"

    if ollama pull "$model"; then
        log_info "Model $model is ready"
    else
        log_error "Failed to pull model: $model"
        exit 1
    fi
}

check_model_exists() {
    local model=$1
    ollama list 2>/dev/null | grep -q "^${model}" && return 0 || return 1
}

main() {
    # Parse arguments
    if [ "$1" = "--check" ]; then
        check_ollama_installed
        if check_ollama_running; then
            log_info "Ollama is running at ${OLLAMA_HOST}"
            ollama list 2>/dev/null || true
            exit 0
        else
            log_warn "Ollama is not running"
            exit 1
        fi
    fi

    # Check if Ollama is installed
    check_ollama_installed

    # Start Ollama if not running
    if check_ollama_running; then
        log_info "Ollama is already running at ${OLLAMA_HOST}"
    else
        start_ollama
    fi

    # Pull embedding model if not present
    if [ -n "$EMBEDDING_MODEL" ]; then
        if check_model_exists "$EMBEDDING_MODEL"; then
            log_info "Embedding model '$EMBEDDING_MODEL' already available"
        else
            pull_model "$EMBEDDING_MODEL"
        fi
    fi

    # Pull LLM model if specified and not present
    if [ -n "$LLM_MODEL" ]; then
        if check_model_exists "$LLM_MODEL"; then
            log_info "LLM model '$LLM_MODEL' already available"
        else
            pull_model "$LLM_MODEL"
        fi
    fi

    echo ""
    log_info "Ollama is ready for use"
    echo ""
    echo "Available models:"
    ollama list 2>/dev/null || echo "  (none)"
    echo ""
    echo "Ollama endpoint: ${OLLAMA_HOST}"
    echo ""
    echo "To pull additional models:"
    echo "  ollama pull llama3.2"
    echo "  ollama pull mistral"
    echo ""
}

main "$@"
