"""searcher 모듈 테스트

실제 ONNX 모델 없이 mock embedder를 사용하여 파이프라인을 검증합니다.
"""

import numpy as np
import pytest

from airgap_kor_search.chunker import Chunk, Document, TextChunker
from airgap_kor_search.config import AppConfig, IndexConfig, ModelConfig, ChunkConfig, SearchConfig
from airgap_kor_search.indexer import Indexer
from airgap_kor_search.searcher import (
    IndexingResult,
    SearchEngine,
    SearchResponse,
    SearchResult,
)


DIM = 128


# ── 테스트용 Mock Embedder ────────────────────────────────


class MockEmbedder:
    """실제 모델 없이 랜덤 벡터를 반환하는 mock embedder"""

    def __init__(self, dim: int = DIM):
        self.embedding_dim = dim
        self._rng = np.random.default_rng(42)
        self._cache: dict[str, np.ndarray] = {}

    def load(self):
        pass

    @property
    def is_loaded(self):
        return True

    def encode(self, texts: list[str] | str, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors = []
        for text in texts:
            if text not in self._cache:
                vec = self._rng.random(self.embedding_dim).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                self._cache[text] = vec
            vectors.append(self._cache[text])

        return np.vstack(vectors)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ── SearchResult 테스트 ───────────────────────────────────


class TestSearchResult:

    def test_score_percent(self):
        result = SearchResult(
            text="테스트", score=0.856, doc_path="/a.txt", chunk_index=0
        )
        assert result.score_percent == 85.6

    def test_to_dict(self):
        result = SearchResult(
            text="테스트", score=0.9, doc_path="/a.txt", chunk_index=0
        )
        d = result.to_dict()
        assert d["text"] == "테스트"
        assert d["score_percent"] == 90.0


class TestSearchResponse:

    def test_to_dict(self):
        response = SearchResponse(
            query="쿼리",
            results=[
                SearchResult(text="결과", score=0.9, doc_path="/a.txt", chunk_index=0)
            ],
            total_found=1,
            elapsed_ms=12.5,
        )
        d = response.to_dict()
        assert d["query"] == "쿼리"
        assert len(d["results"]) == 1
        assert d["elapsed_ms"] == 12.5


class TestIndexingResult:

    def test_to_dict(self):
        result = IndexingResult(
            documents_processed=3,
            chunks_created=15,
            elapsed_sec=2.345,
        )
        d = result.to_dict()
        assert d["documents_processed"] == 3
        assert d["chunks_created"] == 15
        assert d["elapsed_sec"] == 2.35


# ── SearchEngine 통합 테스트 ──────────────────────────────


class TestSearchEngine:
    """Mock embedder를 사용한 SearchEngine 통합 테스트"""

    @pytest.fixture
    def engine(self, tmp_path):
        """테스트용 SearchEngine 생성"""
        config = AppConfig(
            data_dir=tmp_path / "data",
            model=ModelConfig(model_dir=tmp_path / "model", embedding_dim=DIM),
            chunk=ChunkConfig(chunk_size=200, chunk_overlap=0, min_chunk_length=10),
            index=IndexConfig(
                index_path=tmp_path / "data" / "index.faiss",
                db_path=tmp_path / "data" / "meta.db",
            ),
            search=SearchConfig(top_k=5),
        )

        embedder = MockEmbedder(dim=DIM)
        chunker = TextChunker.from_config(config.chunk)
        indexer = Indexer.from_config(config.index, embedding_dim=DIM)

        engine = SearchEngine(
            config=config,
            embedder=embedder,
            chunker=chunker,
            indexer=indexer,
        )
        engine.open()
        yield engine
        engine.close()

    def test_index_text_and_search(self, engine):
        """텍스트 인덱싱 후 검색 파이프라인 테스트"""
        text = "한국어 형태소 분석은 자연어 처리의 기초입니다. " * 10
        result = engine.index_text(text, source="test.txt")

        assert result.documents_processed == 1
        assert result.chunks_created >= 1
        assert result.elapsed_sec > 0

        response = engine.search("형태소 분석")
        assert isinstance(response, SearchResponse)
        assert response.total_found >= 1
        assert response.elapsed_ms > 0

    def test_index_file(self, engine, tmp_path):
        """파일 인덱싱 테스트"""
        doc_file = tmp_path / "test_doc.txt"
        doc_file.write_text(
            "임베딩 모델을 활용한 시맨틱 검색 엔진입니다. " * 10,
            encoding="utf-8",
        )

        result = engine.index_file(doc_file)
        assert result.documents_processed == 1
        assert result.chunks_created >= 1

    def test_index_directory(self, engine, tmp_path):
        """디렉토리 인덱싱 테스트"""
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        for i in range(3):
            (doc_dir / f"doc{i}.txt").write_text(
                f"문서 {i}번의 내용입니다. 충분히 긴 텍스트를 작성합니다. " * 5,
                encoding="utf-8",
            )

        result = engine.index_directory(doc_dir)
        assert result.documents_processed == 3
        assert result.chunks_created >= 3

    def test_index_empty_directory(self, engine, tmp_path):
        """빈 디렉토리 인덱싱"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = engine.index_directory(empty_dir)
        assert result.documents_processed == 0
        assert len(result.errors) > 0

    def test_search_empty_index(self, engine):
        """빈 인덱스 검색"""
        response = engine.search("아무 쿼리")
        assert response.total_found == 0
        assert response.results == []

    def test_search_with_custom_top_k(self, engine):
        """top_k 파라미터 테스트"""
        for i in range(10):
            engine.index_text(
                f"문서 {i}번 내용입니다. 서로 다른 주제를 다루고 있습니다. " * 5,
                source=f"doc{i}.txt",
            )

        response = engine.search("문서 내용", top_k=3)
        assert len(response.results) <= 3

    def test_delete_document(self, engine):
        """문서 삭제 테스트"""
        engine.index_text("삭제할 문서입니다. " * 10, source="delete_me.txt")
        engine.index_text("유지할 문서입니다. " * 10, source="keep_me.txt")

        stats_before = engine.get_stats()

        deleted = engine.delete_document("delete_me.txt")
        assert deleted >= 1

        stats_after = engine.get_stats()
        assert stats_after["total_chunks"] < stats_before["total_chunks"]

    def test_list_documents(self, engine):
        """문서 목록 조회 테스트"""
        engine.index_text("문서 A 내용입니다. " * 10, source="a.txt")
        engine.index_text("문서 B 내용입니다. " * 10, source="b.txt")

        docs = engine.list_documents()
        assert len(docs) == 2
        paths = {d["doc_path"] for d in docs}
        assert "a.txt" in paths
        assert "b.txt" in paths

    def test_get_stats(self, engine):
        """통계 조회 테스트"""
        engine.index_text("통계 테스트용 문서입니다. " * 10, source="stats.txt")

        stats = engine.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_vectors"] >= 1
        assert stats["total_chunks"] >= 1

    def test_not_open_raises(self, tmp_path):
        """엔진 미개방 상태에서 사용 시 에러"""
        config = AppConfig(data_dir=tmp_path / "data")
        engine = SearchEngine(
            config=config,
            embedder=MockEmbedder(),
            chunker=TextChunker(),
            indexer=Indexer.from_config(
                IndexConfig(
                    index_path=tmp_path / "data" / "index.faiss",
                    db_path=tmp_path / "data" / "meta.db",
                ),
                embedding_dim=DIM,
            ),
        )
        with pytest.raises(RuntimeError, match="열려있지 않습니다"):
            engine.search("테스트")