"""LLM-based memory extraction engine.
Replaces the regex-based pipeline.py with structured LLM extraction.
Uses Ollama qwen3.6 36B primary, DeepSeek V4 Pro fallback.
"""

import json
import re
import time
import logging
from typing import Any

import httpx

from ..config import config

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a memory extraction system. Analyze the following conversation/event text and extract structured, reusable memories.

For each memory found, output a JSON object with these fields:
- "type": one of [preference, fact, concept, procedure, decision, question, event, relationship]
- "content": the memory as a clear, standalone sentence in the original language (Chinese or English)
- "importance": 0.0 to 1.0 — how important this memory is for future reference (be conservative; default 0.3, max 0.8 unless clearly critical)
- "confidence": 0.0 to 1.0 — how certain you are this is accurate based on the text
- "keywords": list of 3-5 search keywords in the original language

Memory type definitions:
- "preference": User likes/dislikes, style preferences, workflow habits (e.g., "用户偏好中文回复，不要 emoji")
- "fact": Verifiable information about systems, tools, configurations (e.g., "Qdrant 运行在 Docker 容器中，端口 6333")
- "concept": Abstract understanding, mental models, architectural patterns (e.g., "AI 工具链采用闭环架构：客户端到 AI 执行再回客户端")
- "procedure": Step-by-step workflows, how-to knowledge (e.g., "PPT 构建流程：避免 Google Fonts，用 sed 精确替换")
- "decision": Why a choice was made, trade-off context (e.g., "选择 Qdrant 而非 Chroma 因为 Docker 兼容性")
- "question": Open questions, uncertainties, TODOs (e.g., "是否应该迁移到 MCP 服务器架构？")
- "event": What happened, when, milestone (e.g., "2026-06-18 构建了 43 页 Slidev 幻灯片")
- "relationship": How entities connect (e.g., "OpenClaw 网关在 WeChat 和 Claude Code 之间路由消息")

Only extract memories that are:
1. Non-obvious (not common knowledge anyone would know)
2. Likely to be useful in future conversations
3. Specific enough to be searchable later
4. About the user, their system, their preferences, their work, or their tools

Do NOT extract:
- Trivial conversation filler ("好的", "明白了", "嗯")
- Temporary UI state ("I'm looking at the screen now")
- Generic pleasantries or greetings
- Information that would already be obvious from context
- Purely ephemeral status that changes minute-to-minute

Text to analyze:
{text}

Output as a JSON array. If no meaningful memories found, output an empty array [].
Output ONLY the JSON array, no other text."""


FALLBACK_PROMPT = """Extract key memories from this text. For each meaningful finding, output JSON with fields: type (preference/fact/concept/procedure/decision/question/event/relationship), content (clear standalone sentence), importance (0-1), confidence (0-1), keywords (3-5 words).

Text: {text}

Output: JSON array only."""


async def extract_memories(
    text: str,
    source: str = "unknown",
    source_ref: str | None = None,
    use_fallback: bool = False,
) -> list[dict]:
    """Extract structured memories from raw text using LLM.

    Args:
        text: Raw text to analyze (conversation, code, config, etc.)
        source: Where this text came from (claude, codex, hermes, manual)
        source_ref: Specific reference (session ID, file path)
        use_fallback: Force using DeepSeek fallback instead of Ollama

    Returns:
        List of memory dicts ready for enrichment and storage
    """
    if not text or len(text.strip()) < 20:
        return []

    # Truncate if needed
    max_chars = config.ingestion.limits.max_transcript_chars
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"

    start_time = time.time()
    raw_response = None
    provider_used = "unknown"

    try:
        if use_fallback or config.models.extraction.provider == "deepseek":
            raw_response = await _call_deepseek(text)
            provider_used = "deepseek"
        else:
            raw_response = await _call_ollama(text)
            provider_used = "ollama"
    except Exception as e:
        logger.warning(f"Primary extractor ({provider_used}) failed: {e}")
        # Try fallback
        if provider_used == "ollama":
            try:
                raw_response = await _call_deepseek(text)
                provider_used = "deepseek"
            except Exception as e2:
                logger.error(f"Fallback extractor also failed: {e2}")
                return []

    if not raw_response:
        return []

    memories = _parse_response(raw_response)
    elapsed = (time.time() - start_time) * 1000

    # Enrich each memory with source info
    for m in memories:
        m["source"] = source
        m["source_ref"] = source_ref
        m["extraction_provider"] = provider_used
        m["extraction_latency_ms"] = elapsed

    # Filter by quality thresholds
    min_conf = config.ingestion.quality.min_confidence
    min_imp = config.ingestion.quality.min_importance
    filtered = [
        m for m in memories
        if m.get("confidence", 0) >= min_conf and m.get("importance", 0) >= min_imp
    ]

    logger.info(
        f"Extracted {len(memories)} memories ({len(filtered)} passed quality filter) "
        f"from {source} via {provider_used} in {elapsed:.0f}ms"
    )

    return filtered


async def _call_ollama(text: str) -> str:
    """Call Ollama API for memory extraction.

    Handles both standard and reasoning/thinking models (e.g., qwen3.6).
    For thinking models, reads from message.thinking if message.content is empty.
    """
    cfg = config.models.extraction
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{cfg.base_url}/api/chat",
            json={
                "model": cfg.model,
                "messages": [
                    {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}
                ],
                "stream": False,
                "options": {
                    "temperature": cfg.temperature,
                    "num_predict": max(cfg.max_tokens, 4096),  # thinking models need more
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["message"]
        content = msg.get("content", "")

        # If content is empty, the model might be a thinking variant
        # Try the thinking field as fallback
        if not content or not content.strip():
            thinking = msg.get("thinking", "")
            if thinking and thinking.strip():
                logger.debug("Using thinking content as fallback (thinking model detected)")
                content = thinking

        return content


async def _call_deepseek(text: str) -> str:
    """Call DeepSeek API as fallback for memory extraction."""
    cfg = config.models.extraction
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{cfg.fallback_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {_get_deepseek_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.fallback_model,
                "messages": [
                    {"role": "user", "content": FALLBACK_PROMPT.format(text=text)}
                ],
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _get_deepseek_key() -> str:
    """Get DeepSeek API key from environment or .env file."""
    import os
    # Try multiple sources
    for var in ["ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]:
        val = os.environ.get(var, "")
        if val:
            return val
    # Fallback: try reading from .env file next to config
    try:
        from pathlib import Path
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k in ("ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
                            return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _parse_response(raw: str) -> list[dict]:
    """Parse LLM response into list of memory dicts."""
    # Try to extract JSON array from response
    # The LLM might wrap it in markdown code blocks or add extra text
    cleaned = raw.strip()

    # Remove markdown code fences if present
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?\s*```$', '', cleaned)

    # Try to find JSON array
    match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        memories = json.loads(cleaned)
        if isinstance(memories, list):
            # Validate each item has required fields
            valid = []
            for item in memories:
                if isinstance(item, dict) and "content" in item:
                    # Set defaults for missing fields
                    item.setdefault("type", "fact")
                    item.setdefault("importance", 0.3)
                    item.setdefault("confidence", 0.5)
                    item.setdefault("keywords", [])
                    # Normalize type
                    if item["type"] not in [
                        "preference", "fact", "concept", "procedure",
                        "decision", "question", "event", "relationship",
                    ]:
                        item["type"] = "fact"
                    valid.append(item)
            return valid
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse extraction response: {e}")
        logger.debug(f"Raw response: {raw[:500]}")

    return []


async def extract_single_memory(text: str, memory_type: str = "fact") -> dict | None:
    """Extract a single memory of a specific type (for programmatic use)."""
    memories = await extract_memories(
        text,
        source="manual",
        use_fallback=False,
    )
    # Return first matching type, or first overall
    for m in memories:
        if m["type"] == memory_type:
            return m
    return memories[0] if memories else None
