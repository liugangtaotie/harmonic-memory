#!/bin/bash
# Harmonic Memory — Bootstrap: process all historical data for first Qdrant population
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
API_URL="${HARMONIC_MEMORY_URL:-http://127.0.0.1:18900}"
PYTHON="${PYTHON:-python3}"

echo "=== Harmonic Memory Bootstrap ==="
echo "Started: $(date)"
echo "API: $API_URL"
echo ""

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    sleep 1
done

# ─── Step 1: Process Claude Code transcripts ───
echo ""
echo "━━━ Step 1: Claude Code transcripts ━━━"
CLAUDE_PROJECTS="/c/Users/15321/.claude/projects"
if [ -d "$CLAUDE_PROJECTS" ]; then
    count=0
    for project_dir in "$CLAUDE_PROJECTS"/*/; do
        [ -d "$project_dir" ] || continue
        proj_name=$(basename "$project_dir")

        # Process history.jsonl if exists
        for hist_file in "$project_dir"history.jsonl "$project_dir"*.jsonl; do
            [ -f "$hist_file" ] || continue

            echo "  Processing: $hist_file"

            # Extract last 50 exchanges (user + assistant turns)
            text=$($PYTHON -c "
import json, sys
exchanges = []
try:
    with open('$hist_file', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                entry = json.loads(line)
                role = entry.get('role', '')
                content = entry.get('content', '')
                if isinstance(content, list):
                    content = ' '.join(c.get('text','') for c in content if isinstance(c, dict))
                if role in ('user', 'assistant') and content.strip():
                    exchanges.append(f'{role}: {content[:1000]}')
            except: pass
except: pass
# Take last 50, join
text = '\n'.join(exchanges[-50:])
print(text[:50000])
" 2>/dev/null)

            if [ -n "$text" ] && [ ${#text} -gt 50 ]; then
                # POST to ingest API
                curl -s -X POST "$API_URL/api/v1/ingest" \
                    -H "Content-Type: application/json" \
                    -d "$(jq -n --arg text "$text" --arg src "claude" --arg ref "$hist_file" \
                        '{text: $text, source: $src, source_ref: $ref}')" \
                    > /dev/null 2>&1 || true
                count=$((count + 1))
                echo "    → ingested ($count total)"
            fi

            # Limit to avoid overwhelming
            [ $count -ge 20 ] && break
        done
        [ $count -ge 20 ] && break
    done
    echo "  Claude transcripts processed: $count"
else
    echo "  No Claude projects directory found."
fi

# ─── Step 2: Process existing .md memory files ───
echo ""
echo "━━━ Step 2: Existing markdown memories ━━━"
MEMORY_FILES=$(find /c/Users/15321/.claude/projects -name "*.md" -path "*/memory/*" 2>/dev/null | head -20)
md_count=0
for mdfile in $MEMORY_FILES; do
    [ -f "$mdfile" ] || continue
    echo "  Processing: $mdfile"

    text=$(head -100 "$mdfile" 2>/dev/null)
    if [ -n "$text" ]; then
        curl -s -X POST "$API_URL/api/v1/ingest" \
            -H "Content-Type: application/json" \
            -d "$(jq -n --arg text "$text" --arg src "file" --arg ref "$mdfile" \
                '{text: $text, source: $src, source_ref: $ref}')" \
            > /dev/null 2>&1 || true
        md_count=$((md_count + 1))
        echo "    → ingested"
    fi
done
echo "  Markdown memories processed: $md_count"

# ─── Step 3: Process Codex sessions ───
echo ""
echo "━━━ Step 3: Codex sessions ━━━"
CODEX_SESSIONS="/c/Users/15321/.codex/sessions"
if [ -d "$CODEX_SESSIONS" ]; then
    cx_count=0
    for sess_file in "$CODEX_SESSIONS"/*.jsonl; do
        [ -f "$sess_file" ] || continue

        text=$(head -50 "$sess_file" 2>/dev/null | head -c 20000)
        if [ -n "$text" ] && [ ${#text} -gt 50 ]; then
            curl -s -X POST "$API_URL/api/v1/ingest" \
                -H "Content-Type: application/json" \
                -d "$(jq -n --arg text "$text" --arg src "codex" --arg ref "$sess_file" \
                    '{text: $text, source: $src, source_ref: $ref}')" \
                > /dev/null 2>&1 || true
            cx_count=$((cx_count + 1))
        fi
        [ $cx_count -ge 10 ] && break
    done
    echo "  Codex sessions processed: $cx_count"
else
    echo "  No Codex sessions directory found."
fi

# ─── Step 4: Process Hermes inbox ───
echo ""
echo "━━━ Step 4: Hermes inbox ━━━"
HERMES_INBOX="/c/Users/15321/.hermes/inbox"
if [ -d "$HERMES_INBOX" ]; then
    hm_count=0
    for msg_file in "$HERMES_INBOX"/*.json; do
        [ -f "$msg_file" ] || continue

        text=$(head -c 10000 "$msg_file" 2>/dev/null)
        if [ -n "$text" ]; then
            curl -s -X POST "$API_URL/api/v1/ingest" \
                -H "Content-Type: application/json" \
                -d "$(jq -n --arg text "$text" --arg src "hermes" --arg ref "$msg_file" \
                    '{text: $text, source: $src, source_ref: $ref}')" \
                > /dev/null 2>&1 || true
            hm_count=$((hm_count + 1))
        fi
        [ $hm_count -ge 5 ] && break
    done
    echo "  Hermes inbox processed: $hm_count"
else
    echo "  No Hermes inbox directory found."
fi

# ─── Step 5: Verify results ───
echo ""
echo "━━━ Step 5: Verification ━━━"
echo "API health:"
curl -s "$API_URL/api/v1/health" | python3 -m json.tool 2>/dev/null || echo "  (health check failed)"

echo ""
echo "Qdrant collection:"
curl -s http://localhost:6333/collections/mem0 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (Qdrant check failed)"

echo ""
echo "Memory stats:"
curl -s "$API_URL/api/v1/stats" | python3 -m json.tool 2>/dev/null || echo "  (stats check failed)"

echo ""
echo "=== Bootstrap Complete ==="
echo "Finished: $(date)"
