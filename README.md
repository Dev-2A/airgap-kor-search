# 🔍 airgap-kor-search

에어갭(Air-gapped) 환경을 위한 경량 한국어 문서 검색 엔진

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 특징

- 🔌 **완전 오프라인**: 인터넷 연결 없이 동작
- 🇰🇷 **한국어 특화**: BGE-M3 임베딩 모델 기반 시맨틱 검색
- ⚡ **경량 스택**: ONNX Runtime + FAISS + SQLite (GPU 불필요)
- 📦 **올인원 패키지**: `pip install` 한 번으로 설치 완료
- 🖥️ **CLI + 웹 UI**: 커맨드라인과 브라우저 모두 지원

## 기술 스택

| 구성 요소 | 기술 |
| --- | --- |
| 임베딩 모델 | BGE-M3 (ONNX Runtime, CPU) |
| 벡터 검색 | FAISS (IndexFlatIP) |
| 메타데이터 저장 | SQLite |
| 문서 처리 | txt, md, pdf, docx 지원 |
| CLI | Click + Rich |
| 웹 UI | FastAPI + Vanilla JS |

## 아키텍처

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  문서 입력   │────▶│   청킹      │────▶│  임베딩     │
│ txt/md/pdf/ │     │ 문단/문장    │     │ BGE-M3     │
│ docx        │     │ 분할        │     │ (ONNX)     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐     ┌──────▼──────┐
                    │  검색 결과   │◀────│   인덱서    │
                    │  반환       │     │ FAISS +    │
                    │             │     │ SQLite     │
                    └─────────────┘     └─────────────┘
```

## 설치

### 일반 환경 (인터넷 가능)

```bash
pip install airgap-kor-search
```

### 에어갭 환경

[에어갭 배포 가이드](docs/airgap-deployment.md)를 참고하세요.

## 빠른 시작

### 1. 초기화

```bash
airgap-kor-search init
```

`./airgap_data/` 디렉토리에 설정 파일과 하위 폴더가 생성됩니다.

### 2. 모델 준비

임베딩 모델을 ONNX 형식으로 준비해야 합니다.  
자세한 방법은 [모델 준비 가이드](docs/model-preparation.md)를 참고하세요.

```bash
airgap_data/model/
├── model.onnx
└── tokenizer.json
```

### 3. 문서 인덱싱

```bash
# 디렉토리 전체 인덱싱
airgap-kor-search index ./my-documents/

# 단일 파일 인덱싱
airgap-kor-search index ./report.pdf
```

### 4. 검색

```bash
# CLI 검색
airgap-kor-search search "한국어 형태소 분석 방법"

# 결과 수 지정
airgap-kor-search search "임베딩 모델 비교" -k 10
```

### 5. 웹 UI

```bash
airgap-kor-search serve

# 포트 변경
airgap-kor-search serve -p 9000
```

브라우저에서 `http://127.0.0.1:8000` 접속

## CLI 명령어

| 명령어 | 설명 | 예시 |
| --- | --- | --- |
| `init` | 설정/디렉토리 초기화 | `airgap-kor-search init` |
| `index` | 문서 인덱싱 | `airgap-kor-search index ./docs/` |
| `search` | 시맨틱 검색 | `airgap-kor-search search "쿼리"` |
| `list` | 인덱싱된 문서 목록 | `airgap-kor-search list` |
| `delete` | 인덱스에서 문서 삭제 | `airgap-kor-search delete /path/doc.txt` |
| `stats` | 인덱스 통계 | `airgap-kor-search stats` |
| `serve` | 웹 UI 서버 실행 | `airgap-kor-search serve -p 8080` |

## API 엔드포인트

`airgap-kor-search serve` 실행 후 사용 가능합니다.

| Method | 경로 | 설명 |
| --- | --- | --- |
| GET | `/api/health` | 서버 상태 확인 |
| POST | `/api/search` | 문서 검색 |
| POST | `/api/index/text` | 텍스트 직접 인덱싱 |
| POST | `/api/index/file` | 파일 업로드 인덱싱 |
| GET | `/api/documents` | 인덱싱된 문서 목록 |
| DELETE | `/api/documents` | 문서 삭제 |
| GET | `/api/stats` | 인덱스 통계 |

### 검색 API 예시

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "한국어 형태소 분석", "top_k": 5}'
```

## 설정

`airgap_data/config.json`에서 설정을 변경할 수 있습니다.

```json
{
  "data_dir": "./airgap_data",
  "model": {
    "model_dir": "./airgap_data/model",
    "embedding_dim": 1024,
    "max_seq_length": 512,
    "batch_size": 32
  },
  "chunk": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "min_chunk_length": 50
  },
  "search": {
    "top_k": 5,
    "score_threshold": 0.0
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8000
  }
}
```

## 지원 모델

| 모델 | Pooling | 차원 | 추천 용도 |
| --- | --- | --- | --- |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (기본) | CLS | 1024 | 최고 성능 |
| [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | Mean | 1024 | E5 선호 시 |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Mean | 384 | 저사양 환경 |

Mean Pooling 모델 사용 시 `MeanPoolingEmbedder`를 사용하세요.

## 개발

```bash
# 클론
git clone https://github.com/Dev-2A/airgap-kor-search.git
cd airgap-kor-search

# 가상환경 & 설치
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 테스트
pytest tests/ -v

# 린트
ruff check src/ tests/
```

## 라이선스

MIT License - [LICENSE](LICENSE) 참조
