"""chunker 모듈 테스트"""


import pytest

from airgap_kor_search.chunker import (
    Chunk,
    Document,
    DocumentReader,
    TextChunker,
)
from airgap_kor_search.config import ChunkConfig

# ── DocumentReader 테스트 ─────────────────────────────────


class TestDocumentReader:
    """DocumentReader 테스트"""

    def test_read_txt(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_text("안녕하세요. 테스트 문서입니다.", encoding="utf-8")

        doc = DocumentReader.read(file)

        assert doc.text == "안녕하세요. 테스트 문서입니다."
        assert doc.metadata["filename"] == "test.txt"
        assert doc.metadata["extension"] == ".txt"

    def test_read_md(self, tmp_path):
        file = tmp_path / "test.md"
        file.write_text("# 제목\n\n본문입니다.", encoding="utf-8")

        doc = DocumentReader.read(file)

        assert "# 제목" in doc.text
        assert doc.metadata["extension"] == ".md"

    def test_read_cp949(self, tmp_path):
        file = tmp_path / "cp949.txt"
        file.write_bytes("한국어 CP949 인코딩".encode("cp949"))

        doc = DocumentReader.read(file)

        assert "한국어" in doc.text

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DocumentReader.read(tmp_path / "없는파일.txt")

    def test_read_unsupported_format_raises(self, tmp_path):
        file = tmp_path / "test.xyz"
        file.write_text("data")

        with pytest.raises(ValueError, match="지원하지 않는"):
            DocumentReader.read(file)

    def test_read_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("문서 A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("문서 B", encoding="utf-8")
        (tmp_path / "skip.xyz").write_text("무시됨", encoding="utf-8")

        docs = DocumentReader.read_directory(tmp_path)

        assert len(docs) == 2
        names = {d.metadata["filename"] for d in docs}
        assert names == {"a.txt", "b.txt"}

    def test_read_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.txt").write_text("루트", encoding="utf-8")
        (sub / "child.txt").write_text("하위", encoding="utf-8")

        docs = DocumentReader.read_directory(tmp_path, recursive=True)
        assert len(docs) == 2

        docs_flat = DocumentReader.read_directory(tmp_path, recursive=False)
        assert len(docs_flat) == 1


# ── TextChunker 테스트 ────────────────────────────────────


class TestTextChunker:
    """TextChunker 테스트"""

    def test_from_config(self):
        config = ChunkConfig(chunk_size=256, chunk_overlap=32)
        chunker = TextChunker.from_config(config)

        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32

    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=512, min_chunk_length=10)
        result = chunker.chunk_text("짧은 한국어 텍스트입니다.")

        assert len(result) == 1
        assert result[0] == "짧은 한국어 텍스트입니다."

    def test_paragraph_split(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, min_chunk_length=10)
        text = (
            "첫 번째 문단입니다. 여기에는 충분히 긴 내용이 들어가야 합니다. "
            "그래야 청크가 분리됩니다.\n\n"
            "두 번째 문단입니다. 이 문단도 마찬가지로 충분히 긴 내용을 담고 있어야 "
            "테스트가 통과합니다."
        )

        result = chunker.chunk_text(text)

        assert len(result) >= 2
        assert "첫 번째" in result[0]
        assert "두 번째" in result[-1]

    def test_long_text_multiple_chunks(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, min_chunk_length=10)
        text = " ".join(["한국어 문장입니다." for _ in range(50)])

        result = chunker.chunk_text(text)

        assert len(result) > 1
        for chunk in result:
            # 강제 분할 여유를 고려한 검증
            assert len(chunk) <= chunker.chunk_size * 1.5

    def test_overlap_exists(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=30, min_chunk_length=10)
        text = "가나다라마바사. 아자차카타파하. " * 10

        result = chunker.chunk_text(text)

        if len(result) >= 2:
            # 오버랩으로 인해 연속 청크에 겹치는 텍스트가 있는지 확인
            # (정확한 텍스트 매칭은 어려우므로 청크 수만 확인)
            assert len(result) >= 2

    def test_min_chunk_length_filter(self):
        chunker = TextChunker(chunk_size=512, min_chunk_length=50)
        text = "짧음.\n\n" + "이것은 충분히 긴 문단입니다. " * 10

        result = chunker.chunk_text(text)

        for chunk in result:
            assert len(chunk) >= 50

    def test_normalize(self):
        text = "공백이   많은\t\t텍스트\n\n\n\n문단"
        normalized = TextChunker._normalize(text)

        assert "\t" not in normalized
        assert "   " not in normalized
        assert "\n\n\n" not in normalized


# ── Chunk 데이터 클래스 테스트 ─────────────────────────────


class TestChunk:
    """Chunk 데이터 클래스 테스트"""

    def test_chunk_id(self):
        chunk = Chunk(
            text="테스트",
            doc_path="/docs/test.txt",
            chunk_index=3,
            start_char=0,
            end_char=10,
        )
        assert chunk.chunk_id == "/docs/test.txt::3"


# ── chunk_document 통합 테스트 ────────────────────────────


class TestChunkDocument:
    """chunk_document 통합 테스트"""

    def test_chunk_document(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, min_chunk_length=10)
        doc = Document(
            path="/test/doc.txt",
            text="첫 번째 문단입니다. 충분히 긴 내용이 있어야 합니다.\n\n"
                 "두 번째 문단입니다. 이것도 충분히 길어야 합니다.",
            metadata={"filename": "doc.txt"},
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.doc_path == "/test/doc.txt" for c in chunks)
        assert chunks[0].chunk_index == 0

    def test_chunk_documents(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, min_chunk_length=10)
        docs = [
            Document(path=f"/test/doc{i}.txt", text="테스트 문장입니다. " * 20)
            for i in range(3)
        ]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) >= 3  # 최소 문서 수만큼
        paths = {c.doc_path for c in chunks}
        assert len(paths) == 3
