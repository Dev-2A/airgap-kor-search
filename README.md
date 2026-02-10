# 🔍 airgap-kor-search

에어갭(Air-gapped) 환경을 위한 경량 한국어 문서 검색 엔진

## 특징

- 🔌**완전 오프라인**: 인터넷 연결 없이 동작
- 🇰🇷 **한국어 특화**: 한국어 임베딩 모델 기만 시맨틱 검색
- ⚡**경량 스택**: ONNX Runtime + FAISS + SQLite (GPU 불필요)
- 📦**올인원 패키지**: `pip install` 한 번으로 설치 완료

## 기술 스택

| 구성 요소 | 기술 |
| --- | --- |
| 임베딩 모델 | ONNX Runtime (CPU) |
| 벡터 검색 | FAISS |
| 메타데이터 저장 | SQLite |
| 문서 처리 | 한국어 텍스트 청킹 |
| CLI | Click |
| 웹 UI | FastAPI + Vanilla JS |

## 빠른 시작

```bash
# 문서 인덱싱
airgap-kor-search index ./my-documents/

# 검색
airgap-kor-search search "한국어 형태소 분석 방법"

# 웹 UI 실행
airgap-kor-search serve
```

## 라이선스

MIT License
