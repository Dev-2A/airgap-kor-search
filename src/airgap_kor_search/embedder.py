"""임베딩 모델 로더

ONNX Runtime을 사용하여 한국어 텍스트를 벡터로 변환합니다.
기본 모델: BAAI/bge-m3 (ONNX 변환)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from airgap_kor_search.config import ModelConfig

logger = logging.getLogger(__name__)


class OnnxEmbedder:
    """ONNX Runtime 기반 텍스트 임베딩 모델

    에어갭 환경에서 인터넷 없이 동작하도록 설계되었습니다.
    모델 파일(.onnx)과 토크나이저 파일(tokenizer.json)이
    로컬 디렉토리에 미리 준비되어 있어야 합니다.

    사용 예시:
        >>> embedder = OnnxEmbedder.from_config(model_config)
        >>> vectors = embedder.encode(["한국어 문장입니다."])
        >>> vectors.shape
        (1, 1024)
    """

    def __init__(
        self,
        model_path: Path | str,
        tokenizer_path: Path | str,
        embedding_dim: int = 1024,
        max_seq_length: int = 512,
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.show_progress = show_progress

        self._session = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> "OnnxEmbedder":
        """ModelConfig에서 OnnxEmbedder를 생성합니다."""
        model_dir = Path(config.model_dir)
        return cls(
            model_path=model_dir / "model.onnx",
            tokenizer_path=model_dir / "tokenizer.json",
            embedding_dim=config.embedding_dim,
            max_seq_length=config.max_seq_length,
            batch_size=config.batch_size,
            show_progress=config.show_progress,
        )

    def load(self) -> None:
        """모델과 토크나이저를 메모리에 로드합니다."""
        self._load_tokenizer()
        self._load_model()
        logger.info(
            "임베딩 모델 로드 완료 (dim=%d, max_seq=%d)",
            self.embedding_dim,
            self.max_seq_length,
        )

    def _load_tokenizer(self) -> None:
        """HuggingFace tokenizers 라이브러리로 토크나이저를 로드합니다."""
        from tokenizers import Tokenizer

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"토크나이저 파일을 찾을 수 없습니다: {self.tokenizer_path}\n"
                f"모델 디렉토리에 tokenizer.json 파일을 준비해주세요."
            )

        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.max_seq_length)
        self._tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]",
            length=None,    # 배치 내 최대 길이에 맞춤
        )
        logger.info("토크나이저 로드 완료: %s", self.tokenizer_path)

    def _load_model(self) -> None:
        """ONNX Runtime 세션을 생성합니다."""
        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX 모델 파일을 찾을 수 없습니다: {self.model_path}\n"
                f"모델 디렉토리에 model.onnx 파일을 준비해주세요."
            )

        # CPU 전용 설정 (에어갭 환경 대부분 GPU 없음)
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX 모델 로드 완료: %s", self.model_path)

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._session is not None and self._tokenizer is not None

    def _ensure_loaded(self) -> None:
        """모델이 로드되지 않았으면 자동으로 로드합니다."""
        if not self.is_loaded:
            self.load()

    def _tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        """텍스트 리스트를 토큰화합니다."""
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encodings], dtype=np.int64
        )
        # token_type_ids가 필요한 모델을 위해 0으로 채움
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def _pool_and_normalize(
        self,
        last_hidden_state: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """CLS 풀링 후 L2 정규화를 수행합니다.

        BGE-M3은 [CLS] 토큰의 출력을 임베딩으로 사용합니다.
        다른 모델(E5 등)은 mean pooling을 사용할 수 있으므로
        필요시 이 메서드를 오버라이드하세요.
        """
        # CLS pooling: 첫 번째 토큰의 hidden state
        embeddings = last_hidden_state[:, 0, :]

        # L2 정규화
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)    # 0으로 나누기 방지
        embeddings = embeddings / norms

        return embeddings

    def encode(
        self,
        texts: list[str] | str,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환합니다.

        Args:
            texts: 변환할 텍스트 또는 텍스트 리스트
            batch_size: 배치 크기 (None이면 기본값 사용)

        Returns:
            (N, embedding_dim) 크기의 numpy 배열
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self._tokenize(batch_texts)

            # ONNX 세션에 필요한 입력만 전달
            input_names = {inp.name for inp in self._session.get_inputs()}
            feed = {k: v for k, v in inputs.items() if k in input_names}

            outputs = self._session.run(None, feed)
            last_hidden_state = outputs[0]

            embeddings = self._pool_and_normalize(
                last_hidden_state, inputs["attention_mask"]
            )
            all_embeddings.append(embeddings)

            if self.show_progress and len(texts) > batch_size:
                done = min(i + batch_size, len(texts))
                logger.info("임베딩 진행: %d / %d", done, len(texts))

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """단일 텍스트를 임베딩 벡터로 변환합니다.

        Args:
            text: 변환할 텍스트

        Returns:
            (embedding_dim,) 크기의 1차원 numpy 배열
        """
        return self.encode([text])[0]


class MeanPoolingEmbedder(OnnxEmbedder):
    """Mean Pooling 방식의 임베딩 모델

    multilingual-e5 시리즈 등 mean pooling을 사용하는 모델에 적합합니다.
    """

    def _pool_and_normalize(
        self,
        last_hidden_state: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Mean pooling 후 L2 정규화를 수행합니다."""
        # attention_mask를 3D로 확장
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(
            np.float32
        )

        # 마스크가 적용된 합
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.maximum(sum_mask, 1e-12)

        embeddings = sum_embeddings / sum_mask

        # L2 정규화
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms

        return embeddings
