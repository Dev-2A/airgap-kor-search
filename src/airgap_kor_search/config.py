"""설정 관리 모듈

검색 엔진의 전역 설정(모델 경로, 청크 크기, DB 경로 등)을 관리합니다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ── 기본값 상수 ──────────────────────────────────────────

DEFAULT_DATA_DIR = Path("./airgap_data")
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "model"
DEFAULT_INDEX_PATH = DEFAULT_DATA_DIR / "index.faiss"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "metadata.db"

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_TOP_K = 5

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8007


@dataclass
class ModelConfig:
    """임베딩 모델 관련 설정"""

    model_dir: Path = DEFAULT_MODEL_DIR
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    max_seq_length: int = 512
    batch_size: int = 32
    show_progress: bool = True

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)


@dataclass
class ChunkConfig:
    """문서 청킹 관련 설정"""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_chunk_length: int = 50


@dataclass
class IndexConfig:
    """인덱스 & 메타데이터 저장소 설정"""

    index_path: Path = DEFAULT_INDEX_PATH
    db_path: Path = DEFAULT_DB_PATH
    use_gpu: bool = False

    def __post_init__(self):
        self.index_path = Path(self.index_path)
        self.db_path = Path(self.db_path)


@dataclass
class SearchConfig:
    """검색 관련 설정"""

    top_k: int = DEFAULT_TOP_K
    score_threshold: float = 0.0


@dataclass
class ServerConfig:
    """웹 서버 설정"""

    host: str = DEFAULT_SERVER_HOST
    port: str = DEFAULT_SERVER_PORT


@dataclass
class AppConfig:
    """애플리케이션 전체 설정"""

    data_dir: Path = DEFAULT_DATA_DIR
    model: ModelConfig = field(default_factory=ModelConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)

    def ensure_dirs(self) -> None:
        """필요한 디렉토리를 자동 생성합니다."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model.model_dir.mkdir(parents=True, exist_ok=True)
        self.index.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.db_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환합니다."""
        data = asdict(self)
        # Path 객체를 문자열로 변환
        return _convert_paths(data)

    def save(self, path: Path | str) -> None:
        """설정을 JSON 파일로 저장합니다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "AppConfig":
        """JSON 파일에서 설정을 로드합니다."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """딕셔너리에서 설정 객체를 생성합니다."""
        return cls(
            data_dir=Path(data.get("data_dir", DEFAULT_DATA_DIR)),
            model=ModelConfig(**{
                k: Path(v) if k.endswith("_dir") else v
                for k, v in data.get("model", {}).items()
            }),
            chunk=ChunkConfig(**data.get("chunk", {})),
            index=IndexConfig(**{
                k: Path(v) if k.endswith("_path") else v
                for k, v in data.get("index", {}).items()
            }),
            search=SearchConfig(**data.get("search", {})),
            server=ServerConfig(**data.get("server", {})),
        )


def _convert_paths(obj):
    """딕셔너리 내 Path 객체를 재귀적으로 문자열로 변환합니다."""
    if isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_or_create_config(
    config_path: Optional[Path | str] = None,
) -> AppConfig:
    """설정 파일이 있으면 로드하고, 없으면 기본값으로 생성합니다.

    Args:
        config_path: 설정 파일 경로. None이면 기본 경로 사용.

    Returns:
        AppConfig 인스턴스
    """
    if config_path is None:
        config_path = DEFAULT_DATA_DIR / "config.json"

    config_path = Path(config_path)

    if config_path.exists():
        return AppConfig.load(config_path)

    config = AppConfig()
    config.save(config_path)
    return config
