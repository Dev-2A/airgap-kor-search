"""indexer 모듈 테스트"""

import numpy as np
import pytest

from airgap_kor_search.chunker import Chunk
from airgap_kor_search.config import IndexConfig
from airgap_kor_search.indexer import Indexer, MetadataStore, VectorIndex

DIM = 128  # 테스트용 저차원


def make_chunks(n: int, doc_path: str = "/test/doc.txt") -> list[Chunk]:
    """테스트용 청크 생성"""
    return [
        Chunk(
            text=f"테스트 청크 {i}번 내용입니다.",
            doc_path=doc_path,
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            metadata={"filename": "doc.txt"},
        )
        for i in range(n)
    ]


def make_vectors(n: int, dim: int = DIM) -> np.ndarray:
    """테스트용 정규화 벡터 생성"""
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ── MetadataStore 테스트 ──────────────────────────────────


class TestMetadataStore:

    def test_open_creates_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = MetadataStore(db_path)
        store.open()
        assert db_path.exists()
        assert store.count() == 0
        store.close()

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "test.db"
        with MetadataStore(db_path) as store:
            assert store.count() == 0

    def test_insert_and_get(self, tmp_path):
        chunks = make_chunks(5)
        with MetadataStore(tmp_path / "test.db") as store:
            store.insert_chunks(chunks, start_id=0)

            assert store.count() == 5

            results = store.get_by_ids([0, 2, 4])
            assert len(results) == 3
            assert results[0]["chunk_index"] == 0
            assert results[1]["chunk_index"] == 2

    def test_get_by_doc_path(self, tmp_path):
        chunks_a = make_chunks(3, doc_path="/a.txt")
        chunks_b = make_chunks(2, doc_path="/b.txt")
        with MetadataStore(tmp_path / "test.db") as store:
            store.insert_chunks(chunks_a, start_id=0)
            store.insert_chunks(chunks_b, start_id=3)

            results = store.get_by_doc_path("/a.txt")
            assert len(results) == 3

    def test_delete_by_doc_path(self, tmp_path):
        chunks = make_chunks(5)
        with MetadataStore(tmp_path / "test.db") as store:
            store.insert_chunks(chunks, start_id=0)
            deleted = store.delete_by_doc_path("/test/doc.txt")

            assert len(deleted) == 5
            assert store.count() == 0

    def test_list_documents(self, tmp_path):
        chunks_a = make_chunks(3, doc_path="/a.txt")
        chunks_b = make_chunks(2, doc_path="/b.txt")
        with MetadataStore(tmp_path / "test.db") as store:
            store.insert_chunks(chunks_a, start_id=0)
            store.insert_chunks(chunks_b, start_id=3)

            docs = store.list_documents()
            assert len(docs) == 2


# ── VectorIndex 테스트 ────────────────────────────────────


class TestVectorIndex:

    def test_create_and_add(self, tmp_path):
        idx = VectorIndex(DIM, tmp_path / "test.faiss")
        idx.create()

        vecs = make_vectors(10)
        ids = np.arange(10, dtype=np.int64)
        idx.add(vecs, ids)

        assert idx.total == 10

    def test_save_and_load(self, tmp_path):
        idx = VectorIndex(DIM, tmp_path / "test.faiss")
        idx.create()
        idx.add(make_vectors(5), np.arange(5, dtype=np.int64))
        idx.save()

        idx2 = VectorIndex(DIM, tmp_path / "test.faiss")
        idx2.load()
        assert idx2.total == 5

    def test_load_or_create(self, tmp_path):
        idx = VectorIndex(DIM, tmp_path / "test.faiss")
        idx.load_or_create()  # 파일 없으면 생성
        assert idx.total == 0

    def test_search(self, tmp_path):
        idx = VectorIndex(DIM, tmp_path / "test.faiss")
        idx.create()

        vecs = make_vectors(10)
        ids = np.arange(10, dtype=np.int64)
        idx.add(vecs, ids)

        scores, result_ids = idx.search(vecs[0], top_k=3)
        assert len(scores) == 3
        assert result_ids[0] == 0  # 자기 자신이 가장 유사

    def test_search_empty_index(self, tmp_path):
        idx = VectorIndex(DIM, tmp_path / "test.faiss")
        idx.create()

        query = make_vectors(1)[0]
        scores, ids = idx.search(query, top_k=5)
        assert len(scores) == 0
        assert len(ids) == 0

    def test_remove(self, tmp_path):
        idx = VectorIndex(DIM, tmp_path / "test.faiss")
        idx.create()
        idx.add(make_vectors(10), np.arange(10, dtype=np.int64))

        idx.remove([0, 1, 2])
        assert idx.total == 7

    def test_dimension_mismatch_raises(self, tmp_path):
        idx = VectorIndex(DIM, tmp_path / "test.faiss")
        idx.create()

        wrong_dim_vecs = np.random.rand(3, DIM + 1).astype(np.float32)
        with pytest.raises(ValueError, match="차원 불일치"):
            idx.add(wrong_dim_vecs, np.arange(3, dtype=np.int64))


# ── Indexer 통합 테스트 ───────────────────────────────────


class TestIndexer:

    def test_from_config(self, tmp_path):
        config = IndexConfig(
            index_path=tmp_path / "index.faiss",
            db_path=tmp_path / "meta.db",
        )
        indexer = Indexer.from_config(config, embedding_dim=DIM)
        assert indexer is not None

    def test_add_and_search(self, tmp_path):
        config = IndexConfig(
            index_path=tmp_path / "index.faiss",
            db_path=tmp_path / "meta.db",
        )
        chunks = make_chunks(10)
        vectors = make_vectors(10)

        with Indexer.from_config(config, embedding_dim=DIM) as indexer:
            indexer.add_chunks(chunks, vectors)

            results = indexer.search(vectors[0], top_k=3)
            assert len(results) == 3
            assert results[0]["score"] > 0.9  # 자기 자신

    def test_add_chunks_count_mismatch_raises(self, tmp_path):
        config = IndexConfig(
            index_path=tmp_path / "index.faiss",
            db_path=tmp_path / "meta.db",
        )
        with Indexer.from_config(config, embedding_dim=DIM) as indexer:
            with pytest.raises(ValueError, match="일치하지 않습니다"):
                indexer.add_chunks(make_chunks(5), make_vectors(3))

    def test_delete_document(self, tmp_path):
        config = IndexConfig(
            index_path=tmp_path / "index.faiss",
            db_path=tmp_path / "meta.db",
        )
        chunks = make_chunks(5)
        vectors = make_vectors(5)

        with Indexer.from_config(config, embedding_dim=DIM) as indexer:
            indexer.add_chunks(chunks, vectors)
            assert indexer.total == 5

            deleted = indexer.delete_document("/test/doc.txt")
            assert deleted == 5
            assert indexer.total == 0

    def test_save_and_reopen(self, tmp_path):
        config = IndexConfig(
            index_path=tmp_path / "index.faiss",
            db_path=tmp_path / "meta.db",
        )
        chunks = make_chunks(5)
        vectors = make_vectors(5)

        # 첫 번째 세션: 추가 + 저장
        with Indexer.from_config(config, embedding_dim=DIM) as indexer:
            indexer.add_chunks(chunks, vectors)
            indexer.save()

        # 두 번째 세션: 로드 + 검색
        with Indexer.from_config(config, embedding_dim=DIM) as indexer:
            assert indexer.total == 5
            results = indexer.search(vectors[0], top_k=2)
            assert len(results) == 2

    def test_get_stats(self, tmp_path):
        config = IndexConfig(
            index_path=tmp_path / "index.faiss",
            db_path=tmp_path / "meta.db",
        )
        chunks_a = make_chunks(3, doc_path="/a.txt")
        chunks_b = make_chunks(2, doc_path="/b.txt")
        vectors = make_vectors(5)

        with Indexer.from_config(config, embedding_dim=DIM) as indexer:
            indexer.add_chunks(chunks_a, vectors[:3])
            indexer.add_chunks(chunks_b, vectors[3:])

            stats = indexer.get_stats()
            assert stats["total_vectors"] == 5
            assert stats["total_chunks"] == 5
            assert stats["total_documents"] == 2

    def test_score_threshold(self, tmp_path):
        config = IndexConfig(
            index_path=tmp_path / "index.faiss",
            db_path=tmp_path / "meta.db",
        )
        chunks = make_chunks(10)
        vectors = make_vectors(10)

        with Indexer.from_config(config, embedding_dim=DIM) as indexer:
            indexer.add_chunks(chunks, vectors)

            # 높은 threshold → 결과가 적어져야 함
            results = indexer.search(vectors[0], top_k=10, score_threshold=0.99)
            assert len(results) <= 2
