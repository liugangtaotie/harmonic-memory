#!/bin/bash
# Harmonic Memory — Nightly consolidation trigger
# Invoked by cron: calls the API to run LLM-based memory merging

API_URL="${HARMONIC_MEMORY_URL:-http://127.0.0.1:18900}"

echo "=== Memory Consolidation — $(date) ==="

# Check server is up
if ! curl -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
    echo "ERROR: Memory API not reachable at $API_URL"
    exit 1
fi

# Trigger consolidation (this is a stub — full implementation in Phase 2)
STATS=$(curl -s "$API_URL/api/v1/stats" 2>/dev/null)
TOTAL=$(echo "$STATS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('total_memories',0))" 2>/dev/null)

echo "Current memory count: $TOTAL"
echo "Consolidation will run in Phase 2 (LLM-based merging)"
echo "Done — $(date)"
