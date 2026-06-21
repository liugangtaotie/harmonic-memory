"""Quality and importance scoring for memories."""

import math
from datetime import datetime, timezone


def quality_score(memory: dict) -> float:
    """Calculate composite quality score for a memory.

    Components:
    - confidence: LLM extraction confidence (0-1)
    - source_reliability: how reliable the source is (0-1)
    - specificity: penalize too-short or too-long content
    - freshness: newer = slightly higher base

    Returns 0.0 to 1.0
    """
    # Source reliability weights
    source_weights = {
        "claude": 0.8,   # Direct conversation, most reliable
        "codex": 0.7,    # Code operations
        "hermes": 0.7,   # Agent routing
        "manual": 1.0,   # User explicitly created
        "file": 0.6,     # Auto-extracted from files
        "inference": 0.4,  # System inferred
        "unknown": 0.5,
    }

    components = {}

    # 1. Confidence from extraction (dominant factor)
    components["confidence"] = memory.get("confidence", 0.5)

    # 2. Source reliability
    source = memory.get("source", "unknown")
    components["source"] = source_weights.get(source, 0.5)

    # 3. Specificity (penalize too short or too long)
    content = memory.get("content", "")
    content_len = len(content)
    if content_len < 20:
        specificity = 0.3  # Too short, probably not useful
    elif content_len < 50:
        specificity = 0.6
    elif content_len < 300:
        specificity = 1.0  # Sweet spot
    elif content_len < 500:
        specificity = 0.8
    else:
        specificity = 0.5  # Too long, might be noisy

    components["specificity"] = specificity

    # Weighted average
    weights = {"confidence": 0.5, "source": 0.2, "specificity": 0.3}
    score = sum(components[k] * weights.get(k, 0.33) for k in components)

    return min(1.0, max(0.0, score))


def importance_estimate(content: str, memory_type: str) -> float:
    """Estimate importance based on content signals.

    Returns 0.0 to 1.0
    """
    signals = 0
    score = 0.3  # Base importance

    # Type-based base
    type_bases = {
        "preference": 0.6,
        "decision": 0.7,
        "procedure": 0.5,
        "concept": 0.4,
        "fact": 0.3,
        "relationship": 0.3,
        "event": 0.2,
        "question": 0.2,
    }
    score = type_bases.get(memory_type, 0.3)

    # Content signals
    content_lower = content.lower()

    # Explicit importance markers
    if any(w in content_lower for w in ["重要", "critical", "关键", "必须", "never forget"]):
        score += 0.2
        signals += 1

    # Configuration/path references (usually important for reproducibility)
    if any(w in content_lower for w in ["port", "端口", "api key", "密钥", "password", "config"]):
        score += 0.1
        signals += 1

    # System architecture references
    if any(w in content_lower for w in ["架构", "architecture", "拓扑", "pipeline"]):
        score += 0.1
        signals += 1

    # Personal preference markers
    if any(w in content_lower for w in ["偏好", "prefer", "习惯", "always", "从不"]):
        score += 0.15
        signals += 1

    return min(0.9, score)
