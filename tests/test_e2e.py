"""E2E 테스트

실제 ONNX 모델을 사용한 엔드투엔드 검색 테스트입니다.
모델 파일이 없으면 자동으로 스킵됩니다.

실행 방법:
    1. 모델 준비: python scripts/download_model.py
    2. 테스트 실행: pytest tests/test_e2e.py -v -s
"""

from pathlib import Path

import pytest

from airgap_kor_search.config import (
    AppConfig,
    ChunkConfig,
    IndexConfig,
    ModelConfig,
    SearchConfig,
)
from airgap_kor_search.searcher import SearchEngine


# ── 경로 설정 ─────────────────────────────────────────────

MODEL_DIR = Path("./airgap_data/model")
FIXTURES_DIR = Path(__file__).parent / "fixtures"

HAS_MODEL = (MODEL_DIR / "model.onnx").exists() and (
    MODEL_DIR / "tokenizer.json"
).exists()

skip_no_model = pytest.mark.skipif(
    not HAS_MODEL,
    reason=f"ONNX 모델 없음 ({MODEL_DIR}). "
    "python scripts/download_model.py 로 모델을 준비하세요.",
)


# ── 픽스처 ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """모듈 레벨 SearchEngine (모델 로드가 느리므로 한 번만)"""
    tmp_path = tmp_path_factory.mktemp("e2e")

    config = AppConfig(
        data_dir=tmp_path / "data",
        model=ModelConfig(
            model_dir=MODEL_DIR,
            embedding_dim=1024,
            max_seq_length=512,
            batch_size=8,
        ),
        chunk=ChunkConfig(chunk_size=256, chunk_overlap=32, min_chunk_length=30),
        index=IndexConfig(
            index_path=tmp_path / "data" / "index.faiss",
            db_path=tmp_path / "data" / "meta.db",
        ),
        search=SearchConfig(top_k=5),
    )

    engine = SearchEngine.from_config(config)
    engine.open()

    # 샘플 문서 인덱싱
    if FIXTURES_DIR.exists():
        result = engine.index_directory(FIXTURES_DIR)
        print(
            f"\n📚 E2E 인덱싱: {result.documents_processed}개 문서, "
            f"{result.chunks_created}개 청크 ({result.elapsed_sec:.1f}초)"
        )

    yield engine
    engine.close()


# ── E2E 테스트 ────────────────────────────────────────────


@skip_no_model
class TestE2ESearch:
    """실제 모델을 사용한 검색 테스트"""

    def test_basic_search(self, engine):
        """기본 검색이 결과를 반환하는지"""
        response = engine.search("한국어 형태소 분석")

        assert response.total_found > 0
        print(f"\n🔍 '한국어 형태소 분석' → {response.total_found}건")
        for r in response.results:
            print(f"   [{r.score_percent}%] {r.text[:60]}...")

    def test_semantic_search(self, engine):
        """의미적으로 유사한 결과가 상위에 오는지"""
        response = engine.search("오프라인에서 쓸 수 있는 검색")

        assert response.total_found > 0
        # 에어갭/오프라인/검색 관련 내용이 상위에 와야 함
        top_text = response.results[0].text
        has_relevant = any(
            keyword in top_text
            for keyword in ["에어갭", "오프라인", "인터넷", "차단", "벡터",
                            "검색", "FAISS", "SQLite"]
        )
        print(f"\n🔍 '오프라인에서 쓸 수 있는 검색' → 상위 결과:")
        print(f"   [{response.results[0].score_percent}%] {top_text[:80]}...")
        assert has_relevant, f"관련 없는 결과가 1위: {top_text[:80]}"

    def test_korean_synonym_search(self, engine):
        """한국어 동의어/유사 표현 검색"""
        response = engine.search("문장을 숫자 벡터로 바꾸는 기술")

        assert response.total_found > 0
        # 임베딩 관련 내용이 포함되어야 함
        all_text = " ".join(r.text for r in response.results)
        has_embedding = any(
            keyword in all_text
            for keyword in ["임베딩", "벡터", "변환", "BERT"]
        )
        print(f"\n🔍 '문장을 숫자 벡터로 바꾸는 기술' → {response.total_found}건")
        assert has_embedding

    def test_search_relevance_order(self, engine):
        """검색 결과가 유사도 내림차순인지"""
        response = engine.search("FAISS 벡터 검색 라이브러리")

        if len(response.results) >= 2:
            scores = [r.score for r in response.results]
            assert scores == sorted(scores, reverse=True), "결과가 유사도 순이 아닙니다"

        print(f"\n🔍 점수 순서: {[r.score_percent for r in response.results]}")

    def test_different_queries_different_results(self, engine):
        """다른 쿼리가 다른 결과를 반환하는지"""
        r1 = engine.search("형태소 분석기 종류")
        r2 = engine.search("에어갭 보안 환경")

        if r1.total_found > 0 and r2.total_found > 0:
            top1 = r1.results[0].text[:50]
            top2 = r2.results[0].text[:50]
            # 완전히 같은 결과가 아니어야 함
            assert top1 != top2, "서로 다른 쿼리인데 같은 결과"

        print(f"\n🔍 쿼리1 상위: {r1.results[0].text[:50]}...")
        print(f"🔍 쿼리2 상위: {r2.results[0].text[:50]}...")


@skip_no_model
class TestE2EIndexing:
    """인덱싱 관련 E2E 테스트"""

    def test_stats(self, engine):
        """인덱스 통계가 정상인지"""
        stats = engine.get_stats()

        assert stats["total_documents"] >= 3
        assert stats["total_chunks"] >= 3
        assert stats["total_vectors"] == stats["total_chunks"]

        print(f"\n📊 통계: {stats['total_documents']}문서, "
              f"{stats['total_chunks']}청크, {stats['total_vectors']}벡터")

    def test_list_documents(self, engine):
        """문서 목록이 정상인지"""
        docs = engine.list_documents()

        assert len(docs) >= 3
        for doc in docs:
            assert doc["chunk_count"] > 0

        print(f"\n📋 문서 목록:")
        for doc in docs:
            print(f"   {doc['doc_path']} ({doc['chunk_count']}청크)")

    def test_index_text_directly(self, engine):
        """텍스트 직접 인덱싱 후 검색"""
        engine.index_text(
            "파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다. "
            "데이터 과학, 웹 개발, 인공지능 등 다양한 분야에서 활용됩니다. "
            "풍부한 라이브러리 생태계가 파이썬의 가장 큰 장점입니다.",
            source="e2e_direct_input",
        )

        response = engine.search("파이썬 프로그래밍 장점")
        assert response.total_found > 0

        # 직접 입력한 텍스트가 결과에 포함되어야 함
        sources = [r.doc_path for r in response.results]
        assert "e2e_direct_input" in sources

        print(f"\n🔍 직접 입력 검색: {response.results[0].score_percent}%")

        # 정리
        engine.delete_document("e2e_direct_input")