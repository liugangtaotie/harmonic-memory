#!/bin/bash
# Harmonic Memory — Nightly decay check
# Calculates decay scores and archives memories below threshold

API_URL="${HARMONIC_MEMORY_URL:-http://127.0.0.1:18900}"

echo "=== Memory Decay Check — $(date) ==="

if ! curl -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
    echo "ERROR: Memory API not reachable at $API_URL"
    exit 1
fi

# Run decay via Python (uses the same model as decay.py but against the new system)
python3 -c "
import sys, os
sys.path.insert(0, '/c/Users/15321/harmonic-memory/src')
from lifecycle.decay import run_decay_cycle
import asyncio
asyncio.run(run_decay_cycle())
" 2>&1

echo "Done — $(date)"
