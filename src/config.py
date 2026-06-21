"""Configuration loader for Harmonic Memory."""

import os
from pathlib import Path
from dataclasses import dataclass, field
import yaml


def _expand_path(p: str) -> Path:
    """Expand ~ in paths."""
    return Path(os.path.expanduser(p))


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 18900


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection: str = "mem0"
    vector_size: int = 1536


@dataclass
class SQLiteConfig:
    path: str = "~/.harmonic-memory/memory.db"

    @property
    def expanded_path(self) -> Path:
        return _expand_path(self.path)


@dataclass
class ArchiveConfig:
    path: str = "~/.memoryarchive"

    @property
    def expanded_path(self) -> Path:
        return _expand_path(self.path)


@dataclass
class StorageConfig:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)


@dataclass
class ExtractionModelConfig:
    provider: str = "ollama"
    model: str = "qwen3.6:latest"
    base_url: str = "http://localhost:11434"
    fallback_provider: str = "deepseek"
    fallback_model: str = "deepseek-v4-pro"
    fallback_base_url: str = "https://api.deepseek.com/v1"
    temperature: float = 0.3
    max_tokens: int = 2000


@dataclass
class EmbeddingModelConfig:
    provider: str = "fastembed"
    model: str = "BAAI/bge-small-zh-v1.5"
    base_url: str = "http://localhost:11434"


@dataclass
class ModelsConfig:
    extraction: ExtractionModelConfig = field(default_factory=ExtractionModelConfig)
    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)


@dataclass
class DecayConfig:
    half_life_days: int = 30
    archive_threshold: float = 0.1
    decayed_threshold: float = 0.5
    type_weights: dict = field(default_factory=lambda: {
        "decision": 0.0, "preference": 0.3, "procedure": 0.5,
        "concept": 0.7, "fact": 1.0, "relationship": 1.0,
        "question": 1.5, "event": 2.0,
    })


@dataclass
class ConsolidationConfig:
    schedule: str = "0 2 * * *"
    min_group_size: int = 3
    max_per_group: int = 20


@dataclass
class ProfileConfig:
    update_schedule: str = "0 3 * * *"
    min_confidence: float = 0.6


@dataclass
class LifecycleConfig:
    decay: DecayConfig = field(default_factory=DecayConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


@dataclass
class DedupConfig:
    vector_threshold: float = 0.92


@dataclass
class QualityConfig:
    min_confidence: float = 0.4
    min_importance: float = 0.2


@dataclass
class LimitsConfig:
    max_per_batch: int = 50
    max_transcript_chars: int = 50000
    max_active_memories: int = 2000
    max_db_mb: int = 50
    aggressive_decay_threshold: int = 1500


@dataclass
class IngestionConfig:
    dedup: DedupConfig = field(default_factory=DedupConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)


@dataclass
class HybridSearchConfig:
    keyword_weight: float = 1.0
    vector_weight: float = 0.7
    fusion_k: int = 60


@dataclass
class SearchConfig:
    hybrid: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    max_results: int = 50


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "~/.harmonic-memory/server.log"
    format: str = "json"


@dataclass
class NeuralConfig:
    auto_link_threshold: float = 0.55
    max_auto_links: int = 5
    spread_max_depth: int = 3
    spread_decay_rate: float = 0.6
    spread_max_results: int = 20
    spread_min_activation: float = 0.05


@dataclass
class HarmonicConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)


def load_config(path: str | None = None) -> HarmonicConfig:
    """Load configuration from YAML file, with env var overrides."""
    if path is None:
        # Try multiple locations
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
            _expand_path("~/.harmonic-memory/config.yaml"),
        ]
        for c in candidates:
            if c.exists():
                path = str(c)
                break
        else:
            # No config file found, use defaults
            return HarmonicConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _dict_to_config(raw)


def _dict_to_config(d: dict) -> HarmonicConfig:
    """Recursively convert dict to HarmonicConfig dataclass."""
    cfg = HarmonicConfig()

    if "server" in d:
        cfg.server = ServerConfig(**d["server"])

    if "storage" in d:
        s = d["storage"]
        qdrant_cfg = QdrantConfig(**s.get("qdrant", {}))
        sqlite_cfg = SQLiteConfig(**s.get("sqlite", {}))
        archive_cfg = ArchiveConfig(**s.get("archive", {}))
        cfg.storage = StorageConfig(qdrant=qdrant_cfg, sqlite=sqlite_cfg, archive=archive_cfg)

    if "models" in d:
        m = d["models"]
        ext_cfg = ExtractionModelConfig(**m.get("extraction", {}))
        emb_cfg = EmbeddingModelConfig(**m.get("embedding", {}))
        cfg.models = ModelsConfig(extraction=ext_cfg, embedding=emb_cfg)

    if "lifecycle" in d:
        lc = d["lifecycle"]
        decay_cfg = DecayConfig(**lc.get("decay", {}))
        cons_cfg = ConsolidationConfig(**lc.get("consolidation", {}))
        prof_cfg = ProfileConfig(**lc.get("profile", {}))
        cfg.lifecycle = LifecycleConfig(decay=decay_cfg, consolidation=cons_cfg, profile=prof_cfg)

    if "ingestion" in d:
        ing = d["ingestion"]
        dedup_cfg = DedupConfig(**ing.get("dedup", {}))
        qual_cfg = QualityConfig(**ing.get("quality", {}))
        lim_cfg = LimitsConfig(**ing.get("limits", {}))
        cfg.ingestion = IngestionConfig(dedup=dedup_cfg, quality=qual_cfg, limits=lim_cfg)

    if "search" in d:
        srch = d["search"]
        hyb_cfg = HybridSearchConfig(**srch.get("hybrid", {}))
        cfg.search = SearchConfig(hybrid=hyb_cfg, max_results=srch.get("max_results", 50))

    if "logging" in d:
        cfg.logging = LoggingConfig(**d["logging"])

    if "neural" in d:
        cfg.neural = NeuralConfig(**d["neural"])

    return cfg


# Module-level config singleton
config: HarmonicConfig = load_config()
