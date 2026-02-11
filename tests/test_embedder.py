"""embedder 모듈 테스트

실제 ONNX 모델 파일 없이 mock을 사용하여 로직을 검증합니다.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from airgap_kor_search.config import ModelConfig
from airgap_kor_search.embedder import OnnxEmbedder, MeanPoolingEmbedder


class TestOnnxEmbedder:
    """OnnxEmbedder 기본 동작 테스트"""

    def test_from_config(self, tmp_path):
        config = ModelConfig(model_dir=tmp_path)
        embedder = OnnxEmbedder.from_config(config)

        assert embedder.model_path == tmp_path / "model.onnx"
        assert embedder.tokenizer_path == tmp_path / "tokenizer.json"
        assert embedder.embedding_dim == 1024

    def test_is_loaded_false_initially(self, tmp_path):
        embedder = OnnxEmbedder(
            model_path=tmp_path / "model.onnx",
            tokenizer_path=tmp_path / "tokenizer.json",
        )
        assert embedder.is_loaded is False

    def test_missing_tokenizer_raises(self, tmp_path):
        embedder = OnnxEmbedder(
            model_path=tmp_path / "model.onnx",
            tokenizer_path=tmp_path / "nonexistent.json",
        )
        with pytest.raises(FileNotFoundError, match="토크나이저"):
            embedder._load_tokenizer()

    def test_missing_model_raises(self, tmp_path):
        embedder = OnnxEmbedder(
            model_path=tmp_path / "nonexistent.onnx",
            tokenizer_path=tmp_path / "tokenizer.json",
        )
        with pytest.raises(FileNotFoundError, match="ONNX"):
            embedder._load_model()

    def test_encode_empty_list(self, tmp_path):
        embedder = OnnxEmbedder(
            model_path=tmp_path / "model.onnx",
            tokenizer_path=tmp_path / "tokenizer.json",
            embedding_dim=1024,
        )
        # 빈 리스트는 모델 로드 없이 처리 가능하도록 설계
        embedder._session = MagicMock()  # loaded 상태로 만듦
        embedder._tokenizer = MagicMock()
        result = embedder.encode([])
        assert result.shape == (0, 1024)

    def test_pool_and_normalize_cls(self):
        """CLS 풀링 + L2 정규화 검증"""
        embedder = OnnxEmbedder.__new__(OnnxEmbedder)

        # (batch=2, seq_len=4, dim=3) 가상 hidden state
        hidden = np.array([
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ])
        mask = np.array([[1, 1, 1, 0], [1, 1, 1, 0]])

        result = embedder._pool_and_normalize(hidden, mask)

        assert result.shape == (2, 3)
        # CLS 토큰(index 0)의 값이 정규화되었는지 확인
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
        # 첫 번째 배치: [1,0,0] → 정규화해도 [1,0,0]
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=1e-6)


class TestMeanPoolingEmbedder:
    """MeanPoolingEmbedder 풀링 로직 테스트"""

    def test_mean_pooling(self):
        embedder = MeanPoolingEmbedder.__new__(MeanPoolingEmbedder)

        hidden = np.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        mask = np.array([[1, 1, 0]])  # 세 번째 토큰은 패딩

        result = embedder._pool_and_normalize(hidden, mask)

        assert result.shape == (1, 3)
        # mean of [1,0,0] and [0,1,0] = [0.5, 0.5, 0] → 정규화
        expected = np.array([0.5, 0.5, 0.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result[0], expected, atol=1e-6)