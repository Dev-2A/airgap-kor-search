"""인덱서 모듈

FAISS 벡터 인덱스와 SQLite 메타데이터 저장소를 관리합니다.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from airgap_kor_search.chunker import Chunk
from airgap_kor_search.config import IndexConfig

logger = logging.getLogger(__name__)


# ── SQLite 메타데이터 저장소 ──────────────────────────────


class MetadataStore:
    """SQLite 기반 청크 메타데이터 저장소

    FAISS 인덱스의 정수 ID와 청크 텍스트/메타데이터를 매핑합니다.

    스키마:
        chunks (
            id          INTEGER PRIMARY KEY,  -- FAISS 벡터 ID와 동일
            chunk_id    TEXT UNIQUE,           -- "doc_path::chunk_index"
            text        TEXT,
            doc_path    TEXT,
            chunk_index INTEGER,
            start_char  INTEGER,
            end_char    INTEGER,
            metadata    TEXT                   -- JSON
        )
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        id          INTEGER PRIMARY KEY,
        chunk_id    TEXT UNIQUE NOT NULL,
        text        TEXT NOT NULL,
        doc_path    TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_char  INTEGER NOT NULL,
        end_char    INTEGER NOT NULL,
        metadata    TEXT DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_doc_path ON chunks(doc_path);
    CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id);
    """
    
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
    
    def open(self) -> None:
        """데이터베이스 연결을 열고 스키마를 초기화합니다."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        logger.info("메타데이터 DB 열기 완료: %s", self.db_path)
    
    def close(self) -> None:
        """데이터베이스 연결을 닫습니다."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("데이터베이스가 열려있지 않습니다. open()을 먼저 호출하세요.")
        return self._conn
    
    def insert_chunks(self, chunks: list[Chunk], start_id: int = 0) -> int:
        """청크 메타데이터를 일괄 삽입합니다.
        
        Args:
            chunks: 삽입할 청크 리스트
            start_id: 시작 FAISS 벡터 ID
        
        Returns:
            삽입된 청크 수
        """
        rows = [
            (
                start_id + i,
                chunk.chunk_id,
                chunk.text,
                chunk.doc_path,
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                json.dumps(chunk.metadata, ensure_ascii=False),
                
            )
            for i, chunk in enumerate(chunks)
        ]
        
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (id, chunk_id, text, doc_path, chunk_index, start_char, end_char, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        logger.info("%d개 청크 메타데이터 삽입 완료 (start_id=%d)", len(rows), start_id)
        return len(rows)
    
    def get_by_ids(self, ids: list[int]) -> list[dict]:
        """FAISS ID  목록으로 청크 메타데이터를 조회합니다.
        
        Args:
            ids: FAISS 벡터 ID 리스트
        
        Returns:
            청크 메타데이터 딕셔너리 리스트 (순서 보장)
        """
        if not ids:
            return []
        
        placeholders = ",".join("?" for _ in ids)
        cursor = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
        )
        rows = {row["id"]: dict(row) for row in cursor.fetchall()}
        
        # 입력 순서 보장
        results = []
        for fid in ids:
            if fid in rows:
                row = rows[fid]
                row["metadata"] = json.loads(row["metadata"])
                results.append(row)
        
        return results
    
    def get_by_doc_path(self, doc_path: str) -> list[dict]:
        """문서 경로로 해당 문서의 모든 청크를 조회합니다."""
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE doc_path = ? ORDER BY chunk_index",
            (doc_path,),
        )
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d["metadata"] = json.loads(d["metadata"])
            results.append(d)
        return results
    
    def delete_by_doc_path(self, doc_path: str) -> list[int]:
        """문서 경로에 해당하는 모든 청크를 삭제합니다.
        
        Returns:
            삭제된 청크의 FAISS ID 리스트 (인덱스에서도 제거 필요)
        """
        cursor = self.conn.execute(
            "SELECT id FROM chunks WHERE doc_path = ?", (doc_path,)
        )
        deleted_ids = [row["id"] for row in cursor.fetchall()]
        
        self.conn.execute("DELETE FROM chunks WHERE doc_path = ?", (doc_path,))
        self.conn.commit()
        logger.info("문서 삭제 완료: %s (%d개 청크)", doc_path, len(deleted_ids))
        return deleted_ids
    
    def count(self) -> int:
        """저장된 총 청크 수를 반환합니다."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]
    
    def list_documents(self) -> list[dict]:
        """인덱싱된 문서 목록을 반환합니다."""
        cursor = self.conn.execute(
            """
            SELECT doc_path, COUNT(*) as chunk_count, MIN(id) as first_id
            FROM chunks
            GROUP BY doc_path
            ORDER BY doc_path
            """
        )
        return [dict(row) for row in cursor.fetchall()]


# ── FAISS 벡터 인덱스 ────────────────────────────────────


class VectorIndex:
    """FAISS 기반 벡터 인덱스
    
    IDMap2 + FlatIP (내적 기반) 인덱스를 사용합니다.
    정규화된 벡터에 대해 내적 = 코사인 유사도이므로
    별도의 코사인 변환이 불필요합니다.
    """
    
    def __init__(self, embedding_dim: int, index_path: Path | str):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self._index: Optional[faiss.IndexIDMap2] = None
    
    def create(self) -> None:
        """새 인덱스를 생성합니다."""
        base_index = faiss.IndexFlatIP(self.embedding_dim)
        self._index = faiss.IndexIDMap2(base_index)
        logger.info("새 FAISS 인덱스 생성 (dim=%d)", self.embedding_dim)
    
    def load(self) -> None:
        """디스크에서 인덱스를 로드합니다."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {self.index_path}")
        
        self._index = faiss.read_index(str(self.index_path))
        logger.info(
            "FAISS 인덱스 로드 완료: %s (%d 벡터)",
            self.index_path,
            self._index.ntotal,
        )
    
    def save(self) -> None:
        """인덱스를 디스크에 저장합니다."""
        if self._index is None:
            raise RuntimeError("저장할 인덱스가 없습니다.")
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        logger.info(
            "FAISS 인덱스 저장 완료: %s (%d 벡터)",
            self.index_path,
            self._index.ntotal,
        )
    
    def load_or_create(self) -> None:
        """인덱스 파일이 있으면 로드, 없으면 새로 생성합니다."""
        if self.index_path.exists():
            self.load()
        else:
            self.create()
    
    @property
    def index(self) -> faiss.IndexIDMap2:
        if self._index is None:
            raise RuntimeError("인덱스가 초기화되지 않았습니다.")
        return self._index
    
    @property
    def total(self) -> int:
        """인덱스에 저장된 벡터 수"""
        return self.index.ntotal if self._index else 0
    
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """벡터를 인덱스에 추가합니다.
        
        Args:
            vectors: (N, embedding_dim) 크기의 float32 배열
            ids: (N,) 크기의 int64 배열(FAISS ID)
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"벡터 차원 불일치: 기대 {self.embedding_dim}, 실제 {vectors.shape[1]}"
            )
        
        self.index.add_with_ids(vectors, ids)
        logger.info("%d개 벡터 추가 완료 (총 %d)", len(ids), self.total)
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """가장 유사한 벡터를 검색합니다.
        
        Args:
            query_vector: (embedding_dim,) 또는 (1, embedding_dim) 크기의 쿼리 벡터
            top_k: 반환할 결과 수
        
        Returns:
            (socres, ids) 튜플
            - scores: (top_k,) 유사도 점수 배열
            - ids: (top_k,) FAISS ID 배열
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        
        # 인덱스에 벡터가 없으면 빈 결과 반환
        if self.total == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        
        actual_k = min(top_k, self.total)
        scores, ids = self.index.search(query_vector, actual_k)
        
        # 배치 차원 제거 (단일 쿼리)
        return scores[0], ids[0]
    
    def remove(self, ids: list[int]) -> None:
        """ID에 해당하는 벡터를 인덱스에서 제거합니다."""
        if not ids:
            return
        
        id_array = np.array(ids, dtype=np.int64)
        self.index.remove_ids(id_array)
        logger.info("%d개 벡터 제거 완료 (남은 벡터: %d)", len(ids), self.total)


# ── 통합 인덱서 ──────────────────────────────────────────


class Indexer:
    """FAISS + SQLite를 통합 관리하는 인덱서
    
    사용 예시:
        >>> indexer = Indexer.from_config(index_config, embedding_dim=1024)
        >>> indexer.open()
        >>> indexer.add_chunks(chunks, vectors)
        >>> indexer.save()
        >>> indexer.close()
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        metadata_store: MetadataStore,
    ):
        self.vector_index = vector_index
        self.metadata_store = metadata_store
        self._is_open = False
    
    @classmethod
    def from_config(
        cls,
        config: IndexConfig,
        embedding_dim: int = 1024,
    ) -> "Indexer":
        """IndexConfig에서 Indexer를 생성합니다."""
        return cls(
            vector_index=VectorIndex(embedding_dim, config.index_path),
            metadata_store=MetadataStore(config.db_path),
        )
    
    def open(self) -> None:
        """인덱스와 메타데이터 저장소를 열거나 생성합니다."""
        self.vector_index.load_or_create()
        self.metadata_store.open()
        self._is_open = True
        logger.info(
            "인덱서 열기 완료 (벡터: %d, 청크: %d)",
            self.vector_index.total,
            self.metadata_store.count(),
        )
    
    def close(self) -> None:
        """리소스를 정리합니다."""
        self.metadata_store.close()
        self._is_open = False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def save(self) -> None:
        """인덱스를 디스크에 저장합니다."""
        self.vector_index.save()
    
    @property
    def total(self) -> int:
        return self.vector_index.total
    
    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """청크와 벡터를 인덱스에 추가합니다.
        
        Args:
            chunks: 추가할 청크 리스트
            vectors: (N, embedding_dim) 크기의 벡터 배열
        
        Returns:
            추가된 청크 수
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"청크 수({len(chunks)})와 벡터 수({len(vectors)})가 일치하지 않습니다."
            )
        
        if len(chunks) == 0:
            return 0
        
        # 현재 인덱스의 마지막 ID 다음부터 시작
        start_id = self.vector_index.total
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
        
        # FAISS에 벡터 추가
        self.vector_index.add(vectors, ids)
        
        # SQLite에 메타데이터 추가
        self.metadata_store.insert_chunks(chunks, start_id=start_id)
        
        logger.info(
            "인덱싱 완료: %d개 청크 추가(총 %d)", len(chunks), self.total
        )
        return len(chunks)
    
    def delete_document(self, doc_path: str) -> int:
        """문서를 인덱스에서 완전히 제거합니다.
        
        Args:
            doc_path: 제거할 문서 경로
        
        Returns:
            제거된 청크 수
        """
        deleted_ids = self.metadata_store.delete_by_doc_path(doc_path)
        
        if deleted_ids:
            self.vector_index.remove(deleted_ids)
        
        return len(deleted_ids)
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """쿼리 벡터로 유사한 청크를 검색합니다.
        
        Args:
            query_vector: 쿼리 임베딩 벡터
            top_k: 반환할 최대 결과 수
            score_threshold: 최소 유사도 점수
        
        Returns:
            검색 결과 리스트 (score 내림차순)
            각 항목: {"score": float, "id": int, "text": str, ...}
        """
        scores, ids = self.vector_index.search(query_vector, top_k)
        
        # 유효한 결과만 필터링 (FAISS는 결과 없으면 -1 반환)
        valid_mask = ids >= 0
        if score_threshold > 0:
            valid_mask &= scores >= score_threshold
        
        valid_ids = ids[valid_mask].tolist()
        valid_scores = scores[valid_mask].tolist()
        
        if not valid_ids:
            return []
        
        # 메타데이터 조회
        metadata_list = self.metadata_store.get_by_ids(valid_ids)
        
        # score 병합
        id_to_score = dict(zip(valid_ids, valid_scores))
        results = []
        for meta in metadata_list:
            meta["score"] = id_to_score.get(meta["id"], 0.0)
            results.append(meta)
        
        # score 내림차순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def list_documents(self) -> list[dict]:
        """인덱싱된 문서 목록을 반환합니다."""
        return self.metadata_store.list_documents()
    
    def get_stats(self) -> dict:
        """인덱스 통계를 반환합니다."""
        docs = self.list_documents()
        return {
            "total_vectors": self.vector_index.total,
            "total_chunks": self.metadata_store.count(),
            "total_documents": len(docs),
            "documents": docs,
        }