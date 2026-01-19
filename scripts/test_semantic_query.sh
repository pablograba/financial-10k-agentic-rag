#!/usr/bin/env bash

# Default question when no argument is provided
DEFAULT_QUESTION="What are the main risk factors?"

# Use provided argument or fall back to default
QUESTION="${1:-$DEFAULT_QUESTION}"

# Remove surrounding quotes if user accidentally passed "question here"
QUESTION="${QUESTION%\"}"
QUESTION="${QUESTION#\"}"

# ────────────────────────────────────────────────
# 1. Get embedding from Ollama / nomic-embed-text
# ────────────────────────────────────────────────

EMBEDDING_JSON=$(jq -n \
  --arg model "nomic-embed-text" \
  --arg prompt "$QUESTION" \
  '{model: $model, prompt: $prompt}' |
  curl -s -X POST http://localhost:11434/api/embeddings \
    -H "Content-Type: application/json" \
    -d @- )

# Check if we got a valid embedding
if ! echo "$EMBEDDING_JSON" | jq -e '.embedding' >/dev/null 2>&1; then
    echo "Error: Failed to get embedding from Ollama" >&2
    echo "Response was:" >&2
    echo "$EMBEDDING_JSON" >&2
    exit 1
fi

# Extract compact JSON array of the embedding
VECTOR=$(echo "$EMBEDDING_JSON" | jq -c '.embedding')

# ────────────────────────────────────────────────
# 2. Search Qdrant
# ────────────────────────────────────────────────

curl -s http://localhost:6333/collections/sp500_10k_chunks/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": '"${VECTOR}"',
    "limit": 5,
    "with_payload": true
  }' | jq .

echo ""
echo "(question used: ${QUESTION})"