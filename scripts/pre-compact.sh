#!/bin/bash
# Harmonic Memory — PreCompact hook: inject relevant context before session compaction
# Invoked by Claude Code settings.json PreCompact hook

API_URL="${HARMONIC_MEMORY_URL:-http://127.0.0.1:18900}"

# Read stdin for session context (Claude Code sends JSON)
STDIN_DATA=$(cat 2>/dev/null || echo "{}")

# Extract the current conversation topic/summary for context search
QUERY=$(echo "$STDIN_DATA" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    # Try to get the last user message or session summary
    msgs = data.get('messages', data.get('transcript', []))
    if isinstance(msgs, list) and msgs:
        last_user = ''
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get('role') == 'user':
                last_user = m.get('content', '')
                break
        if last_user:
            print(last_user[:200])
except: pass
" 2>/dev/null)

if [ -z "$QUERY" ]; then
    exit 0
fi

# Search for relevant memories
RESULTS=$(curl -s "$API_URL/api/v1/search?q=$(echo "$QUERY" | python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.stdin.read()))")&limit=5" 2>/dev/null)

# Output as context injection (Claude Code reads stdout)
echo "$RESULTS" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    memories = data.get('results', [])
    if memories:
        print('<!-- Harmonic Memory Context -->')
        for i, m in enumerate(memories[:5]):
            print(f'{i+1}. [{m[\"type\"]}] {m[\"content\"]}')
except: pass
" 2>/dev/null

exit 0
