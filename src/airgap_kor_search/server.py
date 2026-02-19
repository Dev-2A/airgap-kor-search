"""웹 서버 모듈

FastAPI 기반 웹 UI 서버를 제공합니다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from airgap_kor_search.searcher import SearchEngine

logger = logging.getLogger(__name__)

# ── 전역 엔진 참조 ───────────────────────────────────────

_engine = Optional[SearchEngine] = None

WEB_DIR = Path(__file__).parent.parent.parent / "web"


# ── 요청/응답 모델 ───────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0


class IndexTextRequest(BaseModel):
    text: str
    source: str = "<직접 입력>"


class DeleteRequest(BaseModel):
    doc_path: str


class MessageResponse(BaseModel):
    message: str


# ── 앱 팩토리 ────────────────────────────────────────────


def create_app(config_path: Optional[str | Path] = None) -> FastAPI:
    """FastAPI 앱을 생성합니다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """앱 시작/종료 시 엔진을 관리합니다."""
        global _engine
        _engine = SearchEngine.from_config_path(config_path)
        _engine.open()
        logger.info("검색 엔진 시작 완료")
        yield
        _engine.close()
        _engine = None
        logger.info("검색 엔진 종료")

    app = FastAPI(
        title="airgap-kor-search",
        description="에어갭 환경을 위한 경량 한국어 문서 검색 엔진",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── API 라우트 ────────────────────────────────────────

    @app.get("/api/health")
    async def health():
        """서버 상태 확인"""
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/api/search")
    async def api_search(req: SearchRequest):
        """문서 검색"""
        if not _engine:
            raise HTTPException(status_code=503, detail="엔진이 준비되지 않았습니다.")

        if not req.query.strip():
            raise HTTPException(status_code=400, detail="검색어를 입력해주세요.")

        response = _engine.search(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
        )
        return response.to_dict()

    @app.post("/api/index/text")
    async def api_index_text(req: IndexTextRequest):
        """텍스트 직접 인덱싱"""
        if not _engine:
            raise HTTPException(status_code=503, detail="엔진이 준비되지 않았습니다.")

        if not req.text.strip():
            raise HTTPException(status_code=400, detail="텍스트를 입력해주세요.")

        result = _engine.index_text(req.text, source=req.source)
        return result.to_dict()

    @app.post("/api/index/file")
    async def api_index_file(file: UploadFile = File(...)):
        """파일 업로드 인덱싱"""
        if not _engine:
            raise HTTPException(status_code=503, detail="엔진이 준비되지 않았습니다.")

        # 임시 파일로 저장 후 인덱싱
        import tempfile

        suffix = Path(file.filename).suffix if file.filename else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = _engine.index_file(tmp_path)
            # 원본 파일명으로 소스 업데이트
            if file.filename and result.chunks_created > 0:
                # 임시 경로 대신 원본 파일명을 기록하기 위해
                # 인덱싱 후에는 삭제하고 원본 이름으로 재인덱싱
                _engine.delete_document(str(tmp_path))

                from airgap_kor_search.chunker import DocumentReader

                doc = DocumentReader.read(tmp_path)
                doc.path = file.filename
                doc.metadata["filename"] = file.filename

                chunks = _engine.chunker.chunk_document(doc)
                if chunks:
                    vectors = _engine._embed_chunks(chunks)
                    _engine.indexer.add_chunks(chunks, vectors)
                    _engine.indexer.save()
                    result.documents_processed = 1
                    result.chunks_created = len(chunks)

            return result.to_dict()
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/api/documents")
    async def api_list_documents():
        """인덱싱된 문서 목록"""
        if not _engine:
            raise HTTPException(status_code=503, detail="엔진이 준비되지 않았습니다.")

        return {"documents": _engine.list_documents()}

    @app.delete("/api/documents")
    async def api_delete_document(doc_path: str = Query(...)):
        """문서 삭제"""
        if not _engine:
            raise HTTPException(status_code=503, detail="엔진이 준비되지 않았습니다.")

        deleted = _engine.delete_document(doc_path)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

        return {"mesage": f"삭제 완료: {deleted}개 청크", "deleted_chunks": deleted}

    @app.get("/api/stats")
    async def api_stats():
        """인덱스 통계"""
        if not _engine:
            raise HTTPException(status_code=503, detail="엔진이 준비되지 않았습니다.")

        return _engine.get_stats()

    # ── 정적 파일 & 프론트엔드 ────────────────────────────

    # web/ 디렉토리가 있으면 정적 파일 서빙
    web_dir = WEB_DIR
    if not web_dir.exists():
        # 패키지 설치 시 경로가 다를 수 있으므로 대체 경로도 시도
        alt_web_dir = Path(__file__).parent / "web"
        if alt_web_dir.exists():
            web_dir = alt_web_dir

    if web_dir.exists():
        @app.get("/", response_class=HTMLResponse)
        async def serve_index():
            index_html = web_dir / "index.html"
            return index_html.read_text(encoding="utf-8")

        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    return app
