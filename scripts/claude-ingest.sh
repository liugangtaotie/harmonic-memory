#!/bin/bash
# Harmonic Memory — Claude Code Stop hook
# Reads stdin JSON from Claude Code (session metadata),
# extracts the last exchanges from the transcript,
# and POSTs to the memory API for async ingestion.
#
# Non-blocking: sends HTTP request, does NOT wait for extraction.

API_URL="${HARMONIC_MEMORY_URL:-http://127.0.0.1:18900}"
PYTHON="/f/aiAgent/harmonic-memory/.venv/Scripts/python"
TEMP_DIR="${HOME}/.harmonic-memory/tmp"
TEMP_DIR_WIN=$(cygpath -w "$TEMP_DIR" 2>/dev/null || echo "$TEMP_DIR")
mkdir -p "$TEMP_DIR"

# ── Step 1: Find the transcript ──
TRANSCRIPT=""

# Try reading stdin (Claude Code passes session JSON)
STDIN_DATA=$(cat 2>/dev/null || echo "")
if [ -n "$STDIN_DATA" ]; then
    TRANSCRIPT=$(echo "$STDIN_DATA" | "$PYTHON" -c "
import json,sys
try:
    data = json.loads(sys.stdin.read())
    # Try multiple possible field names
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

# Nothing to process
if [ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ]; then
    exit 0
fi

# Convert Unix path to Windows for Python (cygpath available in Git Bash)
TRANSCRIPT_WIN=$(cygpath -w "$TRANSCRIPT" 2>/dev/null || echo "$TRANSCRIPT")

# ── Step 2: Extract exchanges, chunk, and POST in batches ──
PAYLOAD_FILE="$TEMP_DIR/ingest_payload_$$.json"
PAYLOAD_FILE_WIN="$TEMP_DIR_WIN\\ingest_payload_$$.json"

# Use Python to extract ALL exchanges, chunk by 6000 chars, POST each chunk
"$PYTHON" -c "
import json, sys, os, httpx, time

transcript_path = os.path.expanduser(r'$TRANSCRIPT_WIN')

# Fallback: find latest transcript
if not os.path.exists(transcript_path):
    import glob
    candidates = glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl'))
    if candidates:
        transcript_path = max(candidates, key=os.path.getmtime)

if not os.path.exists(transcript_path):
    sys.exit(0)

# Extract all user/assistant exchanges (not just last 200 lines)
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
                    # Shorter per-exchange limit to fit more in context
                    exchanges.append(f'{role}: {content[:800]}')
            except: pass
except Exception as e:
    pass

if not exchanges:
    sys.exit(0)

# Skip short/repetitive sessions (< 15 exchanges) to avoid ingesting monitoring noise
if len(exchanges) < 15:
    sys.exit(0)

# Use last 200 exchanges, chunk into ~6000 char pieces for reliable extraction
selected = exchanges[-200:]
chunks = []
current = []
current_len = 0
for ex in selected:
    if current_len + len(ex) > 6000 and current:
        chunks.append('\n'.join(current))
        current = []
        current_len = 0
    current.append(ex)
    current_len += len(ex)
if current:
    chunks.append('\n'.join(current))

# Fire-and-forget POST each chunk
api_url = os.environ.get('HARMONIC_MEMORY_URL', 'http://127.0.0.1:18900')
for i, chunk in enumerate(chunks):
    if len(chunk.strip()) < 20:
        continue
    try:
        httpx.post(
            f'{api_url}/api/v1/ingest',
            json={'text': chunk, 'source': 'claude', 'source_ref': f'{transcript_path}#chunk{i}'},
            timeout=300.0
        )
    except:
        pass  # Best-effort, don't block Claude shutdown
"

exit 0
