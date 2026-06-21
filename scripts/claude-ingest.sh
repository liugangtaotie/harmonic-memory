#!/bin/bash
# Harmonic Memory — Claude Code Stop hook
# Extracts recent exchanges and POSTs to the memory API.
# Non-blocking: fire-and-forget, 10s timeout from hook config.
#
# Cooldown: skips if last ingest was < 30 min ago, to avoid
# re-extracting the same growing session file on every stop.

API_URL="${HARMONIC_MEMORY_URL:-http://127.0.0.1:18900}"
PYTHON="/f/harmonic-memory/.venv/Scripts/python"
COOLDOWN_FILE="${HOME}/.harmonic-memory/last_ingest_time"
COOLDOWN_SEC=1800  # 30 minutes

# Fallback if venv doesn't exist yet
if [ ! -f "$PYTHON" ]; then
    PYTHON="/f/aiAgent/harmonic-memory/.venv/Scripts/python"
fi

# ── Cooldown check ──
NOW=$(date +%s)
if [ -f "$COOLDOWN_FILE" ]; then
    LAST=$(cat "$COOLDOWN_FILE" 2>/dev/null || echo 0)
    ELAPSED=$((NOW - LAST))
    if [ "$ELAPSED" -lt "$COOLDOWN_SEC" ]; then
        exit 0  # Too soon, skip
    fi
fi

TEMP_DIR="${HOME}/.harmonic-memory/tmp"
TEMP_DIR_WIN=$(cygpath -w "$TEMP_DIR" 2>/dev/null || echo "$TEMP_DIR")
mkdir -p "$TEMP_DIR"

# ── Step 1: Find the transcript ──
TRANSCRIPT=""
STDIN_DATA=$(cat 2>/dev/null || echo "")
if [ -n "$STDIN_DATA" ]; then
    TRANSCRIPT=$(echo "$STDIN_DATA" | "$PYTHON" -c "
import json,sys
try:
    data = json.loads(sys.stdin.read())
    for key in ('transcript_path','history_path','session_path','path'):
        if key in data:
            print(data[key])
            break
except: pass
" 2>/dev/null)
fi

# Fallback: find most recently modified history file
if [ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ]; then
    TRANSCRIPT=$(find "${HOME}/.claude/projects" -name "*.jsonl" -type f 2>/dev/null | \
        xargs ls -t 2>/dev/null | head -1)
fi

if [ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ]; then
    exit 0
fi

TRANSCRIPT_WIN=$(cygpath -w "$TRANSCRIPT" 2>/dev/null || echo "$TRANSCRIPT")

# ── Step 2: Extract recent exchanges, POST single chunk ──
"$PYTHON" -c "
import json, sys, os, httpx

transcript_path = os.path.expanduser(r'$TRANSCRIPT_WIN')
if not os.path.exists(transcript_path):
    import glob
    candidates = glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl'))
    if candidates:
        transcript_path = max(candidates, key=os.path.getmtime)
if not os.path.exists(transcript_path):
    sys.exit(0)

exchanges = []
try:
    with open(transcript_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                entry = json.loads(line)
                msg = entry.get('message', {})
                role = msg.get('role', '') or entry.get('type', '')
                content = msg.get('content', '')
                if isinstance(content, list):
                    content = ' '.join(
                        c.get('text', '') for c in content
                        if isinstance(c, dict) and c.get('type') == 'text'
                    )
                if role in ('user', 'assistant') and content and content.strip():
                    exchanges.append(f'{role}: {content[:600]}')
            except: pass
except: pass

if len(exchanges) < 10:
    sys.exit(0)

# Only last 60 exchanges (not 200) to focus on the recent conversation
selected = exchanges[-60:]
text = '\n'.join(selected)
if len(text.strip()) < 200:
    sys.exit(0)

# Single POST (not chunked — one ingestion per session stop)
api_url = os.environ.get('HARMONIC_MEMORY_URL', 'http://127.0.0.1:18900')
try:
    httpx.post(
        f'{api_url}/api/v1/ingest',
        json={'text': text, 'source': 'claude', 'source_ref': os.path.basename(transcript_path)},
        timeout=10.0
    )
except: pass
"

# Record cooldown timestamp
echo "$NOW" > "$COOLDOWN_FILE"
exit 0
