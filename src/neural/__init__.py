"""Neural network operations — spreading activation over the memory graph.

Spreading activation is a cognitive model where recalling one memory
triggers related memories through associative links (edges). Activation
flows through the graph, decaying with each hop, similar to how
biological neurons fire together.
"""

import logging
from ..db.sqlite import MemoryDB
from ..config import config

logger = logging.getLogger(__name__)


def spread_activate(
    seed_id: str,
    db: MemoryDB,
    max_depth: int | None = None,
    decay_rate: float | None = None,
    max_results: int | None = None,
    min_activation: float | None = None,
    bidirectional: bool = True,
) -> list[dict]:
    """Spreading activation from a seed memory through the edge graph.

    BFS-based traversal: each hop multiplies activation by
    decay_rate * edge_weight. Bidirectional by default (follows both
    outbound and inbound edges).

    Args:
        seed_id: Starting memory ID
        db: Database connection
        max_depth: How many hops to traverse (default from config)
        decay_rate: Multiplier per hop, 0-1 (default from config)
        max_results: Max memories to return (default from config)
        min_activation: Minimum activation score to include
        bidirectional: Follow both out and in edges

    Returns:
        List of {memory_id, activation, depth, path} sorted by activation desc
    """
    if max_depth is None:
        max_depth = config.neural.spread_max_depth
    if decay_rate is None:
        decay_rate = config.neural.spread_decay_rate
    if max_results is None:
        max_results = config.neural.spread_max_results
    if min_activation is None:
        min_activation = config.neural.spread_min_activation

    # BFS with activation decay
    # Each entry in activation: {memory_id: (total_activation, depth, path)}
    activation: dict[str, float] = {seed_id: 1.0}
    depths: dict[str, int] = {seed_id: 0}
    paths: dict[str, list[str]] = {seed_id: [seed_id]}
    frontier: list[tuple[str, float, int]] = [(seed_id, 1.0, 0)]

    while frontier:
        current_id, current_act, depth = frontier.pop(0)

        if depth >= max_depth:
            continue

        # Get edges from current node
        edges = db.get_edges(current_id, direction="out")
        if bidirectional:
            in_edges = db.get_edges(current_id, direction="in")
            # Normalize in-edges to (source=neighbor, target=current)
            for e in in_edges:
                e = dict(e)
                e["target_id"], e["source_id"] = e["source_id"], current_id
                edges.append(e)

        for edge in edges:
            neighbor = edge["target_id"]
            if neighbor == current_id or neighbor == seed_id:
                continue

            edge_weight = edge.get("weight", 1.0)
            new_activation = current_act * decay_rate * edge_weight

            if new_activation < min_activation:
                continue

            if neighbor not in activation or new_activation > activation[neighbor]:
                activation[neighbor] = new_activation
                depths[neighbor] = depth + 1
                paths[neighbor] = paths.get(current_id, [current_id]) + [neighbor]
                frontier.append((neighbor, new_activation, depth + 1))

    # Remove seed, sort by activation descending
    results = [
        {
            "memory_id": mid,
            "activation": round(act, 4),
            "depth": depths.get(mid, 0),
            "path": paths.get(mid, []),
        }
        for mid, act in activation.items()
        if mid != seed_id
    ]
    results.sort(key=lambda r: r["activation"], reverse=True)

    return results[:max_results]


def reinforce_edges(
    db: MemoryDB,
    activated_ids: list[str],
    boost: float = 0.05,
):
    """Hebbian reinforcement: strengthen edges along an activation path.

    Memories accessed together get stronger connections.
    Called after spreading activation to reinforce the paths taken.
    """
    for i in range(len(activated_ids) - 1):
        src, tgt = activated_ids[i], activated_ids[i + 1]
        # Check if edge exists
        existing = db.conn.execute(
            """SELECT id, weight FROM memory_edges
               WHERE (source_id=? AND target_id=?)
                  OR (source_id=? AND target_id=?)""",
            (src, tgt, tgt, src),
        ).fetchone()

        if existing:
            new_weight = min(1.0, existing["weight"] + boost)
            db.conn.execute(
                "UPDATE memory_edges SET weight=? WHERE id=?",
                (new_weight, existing["id"]),
            )
        else:
            db.add_edge(src, tgt, relation_type="references", weight=boost)

    db.conn.commit()
