"""검색 엔진 코어

쿼리 임베딩 → FAISS 검색 → 메타데이터 조회 → 결과 반환 파이프라인을 구현합니다.
모든 하위 모듈을 통합하여 단일 인터페이스를 제공합니다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from airgap_kor_search.chunker import Chunk, DocumentReader, TextChunker
from airgap_kor_search.config import AppConfig, load_or_create_config
from airgap_kor_search.embedder import OnnxEmbedder
from airgap_kor_search.indexer import Indexer

logger = logging.getLogger(__name__)


# ── 검색 결과 데이터 클래스 ───────────────────────────────


@dataclass
class SearchResult:
    """단일 검색 결과"""

    text: str
    score: float
    doc_path: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def score_percent(self) -> float:
        """유사도 점수를 퍼센트로 반환합니다."""
        return round(self.score * 100, 1)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "score_percent": self.score_percent,
            "doc_path": self.doc_path,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


@dataclass
class SearchResponse:
    """검색 응답 (결과 + 메타 정보)"""

    query: str
    results: list[SearchResult]
    total_found: int
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class IndexingResult:
    """인덱싱 결과"""
    documents_processed: int
    chunks_created: int
    elapsed_sec: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "documents_processed": self.documents_processed,
            "chunks_created": self.chunks_created,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "errors": self.errors,
        }


# ── 검색 엔진 ────────────────────────────────────────────


class SearchEngine:
    """에어갭 한국어 문서 검색 엔진

    사용 예시:
        >>> engine = SearchEngine.from_config_path("./config.json")
        >>> engine.open()
        >>>
        >>> # 문서 인덱싱
        >>> result = engine.index_directory("./documents/")
        print(f"{result.chunks_created}개 청크 인덱싱 완료")
        >>>
        >>> # 검색
        >>> response = engine.search("한국어 형태소 분석")
        >>> for r in response.results:
        ...     print(f"[{r.score_percent}%] {r.text[:80]}...")
        >>>
        >>> engine.close()
    """

    def __init__(
        self,
        config: AppConfig,
        embedder: OnnxEmbedder,
        chunker: TextChunker,
        indexer: Indexer,
    ):
        self.config = config
        self.embedder = embedder
        self.chunker = chunker
        self.indexer = indexer
        self._is_open = False

    @classmethod
    def from_config(cls, config: AppConfig) -> "SearchEngine":
        """AppConfig에서 SearchEngine을 생성합니다."""
        embedder = OnnxEmbedder.from_config(config.model)
        chunker = TextChunker.from_config(config.chunk)
        indexer = Indexer.from_config(config.index, embedding_dim=config.model.embedding_dim)

        return cls(
            config=config,
            embedder=embedder,
            chunker=chunker,
            indexer=indexer,
        )

    @classmethod
    def from_config_path(cls, config_path: Optional[str | Path] = None) -> "SearchEngine":
        """설정 파일 경로에서 SearchEngine을 생성합니다."""
        config = load_or_create_config(config_path)
        return cls.from_config(config)

    def open(self) -> None:
        """엔진을 초기화합니다. (인덱스 로드, 모델 로드)"""
        self.config.ensure_dirs()
        self.indexer.open()
        # 임베딩 모델은 첫 사용 시 지연 로드 (lazy load)
        self._is_open = True
        logger.info("검색 엔진 시작 완료")

    def close(self) -> None:
        """리소스를 정리합니다."""
        self.indexer.close()
        self._is_open = False
        logger.info("검색 엔진 종료")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("엔진이 열려있지 않습니다. open()을 먼저 호출하세요.")

    # ── 인덱싱 ────────────────────────────────────────────

    def index_file(self, file_path: str | Path) -> IndexingResult:
        """단일 파일을 인덱싱합니다."""
        self._ensure_open()
        start = time.time()
        errors = []

        try:
            doc = DocumentReader.read(file_path)
            chunks = self.chunker.chunk_document(doc)

            if not chunks:
                return IndexingResult(
                    documents_processed=1,
                    chunks_created=0,
                    elapsed_sec=time.time() - start,
                    errors=["청크가 생성되지 않았습니다."],
                )

            vectors = self._embed_chunks(chunks)
            self.indexer.add_chunks(chunks, vectors)
            self.indexer.save()

        except Exception as e:
            logger.error("인덱싱 실패 (%s): %s", file_path, e)
            errors.append(f"{file_path}: {e}")
            return IndexingResult(
                documents_processed=0,
                chunks_created=0,
                elapsed_sec=time.time() - start,
                errors=errors,
            )

        return IndexingResult(
            documents_processed=1,
            chunks_created=len(chunks),
            elapsed_sec=time.time() - start,
        )

    def index_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> IndexingResult:
        """디렉토리 내 모든 문서를 인덱싱합니다."""
        self._ensure_open()
        start = time.time()
        errors = []

        try:
            documents = DocumentReader.read_directory(directory, recursive=recursive)
        except Exception as e:
            return IndexingResult(
                documents_processed=0,
                chunks_created=0,
                elapsed_sec=time.time() - start,
                errors=[str(e)],
            )

        if not documents:
            return IndexingResult(
                documents_processed=0,
                chunks_created=0,
                elapsed_sec=time.time() - start,
                errors=["지원하는 문서를 찾을 수 없습니다."],
            )

        all_chunks = self.chunker.chunk_documents(documents)

        if not all_chunks:
            return IndexingResult(
                documents_processed=len(documents),
                chunks_created=0,
                elapsed_sec=time.time() - start,
                errors=["청크가 생성되지 않았습니다."],
            )

        vectors = self._embed_chunks(all_chunks)
        self.indexer.add_chunks(all_chunks, vectors)
        self.indexer.save()

        return IndexingResult(
            documents_processed=len(documents),
            chunks_created=len(all_chunks),
            elapsed_sec=time.time() - start,
            errors=errors,
        )

    def index_text(self, text: str, source: str = "<직접 입력>") -> IndexingResult:
        """텍스트를 직접 인덱싱합니다."""
        self._ensure_open()
        start = time.time()

        from airgap_kor_search.chunker import Document

        doc = Document(path=source, text=text, metadata={"filename": source})
        chunks = self.chunker.chunk_document(doc)

        if not chunks:
            return IndexingResult(
                documents_processed=1,
                chunks_created=0,
                elapsed_sec=time.time() - start,
                errors=["청크가 생성되지 않았습니다."],
            )

        vectors = self._embed_chunks(chunks)
        self.indexer.add_chunks(chunks, vectors)
        self.indexer.save()

        return IndexingResult(
            documents_processed=1,
            chunks_created=len(chunks),
            elapsed_sec=time.time() - start,
        )

    def _embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """청크 리스트를 임베딩합니다."""
        texts = [chunk.text for chunk in chunks]
        return self.embedder.encode(texts)

    # ── 검색 ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> SearchResponse:
        """쿼리 텍스트로 유사한 문서 청크를 검색합니다.

        Args:
            query: 검색 쿼리 텍스트
            top_k: 반환할 최대 결과 수 (None이면 설정값 사용)
            score_threshold: 최소 유사도 점수 (None이면 설정값 사용)

        Returns:
            SearchResponse 객체
        """
        self._ensure_open()
        start = time.time()

        top_k = top_k or self.config.search.top_k
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.search.score_threshold
        )

        # 쿼리 임베딩
        query_vector = self.embedder.encode_single(query)

        # FAISS 검색 + 메타데이터 조회
        raw_results = self.indexer.search(
            query_vector, top_k=top_k, score_threshold=score_threshold
        )

        # SearchResult 변환
        results = [
            SearchResult(
                text=r["text"],
                score=r["score"],
                doc_path=r["doc_path"],
                chunk_index=r["chunk_index"],
                metadata=r.get("metadata", {}),
            )
            for r in raw_results
        ]

        elapsed_ms = (time.time() - start) * 1000

        response = SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            elapsed_ms=elapsed_ms,
        )

        logger.info(
            "검색 완료: '%s' → %d건 (%.1fms)",
            query,
            len(results),
            elapsed_ms,
        )
        return response

    # ── 관리 ──────────────────────────────────────────────

    def delete_document(self, doc_path: str) -> int:
        """인덱스에서 문서를 삭제합니다."""
        self._ensure_open()
        count = self.indexer.delete_document(doc_path)
        if count > 0:
            self.indexer.save()
        return count

    def get_stats(self) -> dict:
        """인덱스 통계를 반환합니다."""
        self._ensure_open()
        return self.indexer.get_stats()

    def list_documents(self) -> list[dict]:
        """인덱싱된 문서 목록을 반환합니다."""
        self._ensure_open()
        return self.indexer.list_documents()
