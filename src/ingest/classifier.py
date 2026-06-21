"""Auto-classify memory type based on content patterns and LLM hints."""

import re

# Heuristic rules for fast classification without LLM
TYPE_PATTERNS: dict[str, list[str]] = {
    "preference": [
        r"(?:偏好|喜欢|prefer|不喜欢|讨厌|习惯|总是|从不|风格|style)",
        r"(?:want|need|必须|应该|should|must|always|never).*(?:reply|回答|respond|use|用)",
    ],
    "procedure": [
        r"(?:步骤|流程|step|procedure|how.?to|构建|build|deploy|部署|配置|configure)",
        r"(?:第一步|第二步|step \d|1\.\s|2\.\s|3\.\s)",
        r"(?:命令|command|运行|run|执行|execute|操作)",
    ],
    "decision": [
        r"(?:选择|choice|chose|为什么|why|因为|because|权衡|trade.?off|决策)",
        r"(?:替代|alternative|instead|rather|而非|而非)",
    ],
    "question": [
        r"(?:\?|？|是否|应该|how|what|when|where|why|TODO|FIXME|待定|pending)",
        r"(?:需要确认|待研究|需要调查|需要了解)",
    ],
    "event": [
        r"(?:完成了|built|created|finished|完成|做了|happened|发生)",
        r"(?:\d{4}-\d{2}-\d{2}|昨天|今天|上周|本月)",
    ],
    "relationship": [
        r"(?:连接|connect|route|路由|between|之间|integrate|集成|桥接|bridge)",
        r"(?:A→B|A→B|→|depends on|依赖于|调用|calls)",
    ],
    "concept": [
        r"(?:架构|architecture|模式|pattern|模型|model|设计|design|思想|philosophy)",
        r"(?:抽象|abstract|概念|concept|原则|principle|理念)",
    ],
    "fact": [
        # Default catch-all — anything that doesn't match above
        r"(?:配置|端口|port|版本|version|路径|path|API|key|密钥|地址|url)",
        r"(?:安装|install|运行|running|状态|status|内存|memory|进程|process)",
    ],
}


def classify(content: str, llm_type_hint: str | None = None) -> str:
    """Classify memory type from content.

    Args:
        content: The memory content text
        llm_type_hint: Type suggested by the LLM extractor (trusted if present)

    Returns:
        Memory type string
    """
    # Trust LLM classification if provided
    valid_types = {
        "preference", "fact", "concept", "procedure",
        "decision", "question", "event", "relationship",
    }
    if llm_type_hint and llm_type_hint in valid_types:
        return llm_type_hint

    # Heuristic scoring
    scores: dict[str, int] = {}
    for mem_type, patterns in TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 1
        scores[mem_type] = score

    # Return highest scoring type, default to "fact"
    if not scores or max(scores.values()) == 0:
        return "fact"

    return max(scores, key=scores.get)


def classify_batch(memories: list[dict]) -> list[dict]:
    """Classify a batch of memories in-place."""
    for m in memories:
        llm_hint = m.pop("type", None) if "type" in m else None
        m["type"] = classify(m.get("content", ""), llm_hint)
    return memories
