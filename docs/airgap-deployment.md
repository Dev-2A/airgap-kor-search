# 🚀 에어갭 환경 배포 가이드

인터넷이 차단된 에어갭(Air-gapped) 환경에 airgap-kor-search를 배포하는 방법입니다.

## 개요

에어갭 배포는 두 단계로 진행됩니다:

1. **인터넷 환경 (준비 단계)**: 패키지, 모델 파일을 다운로드
2. **에어갭 환경 (설치 단계)**: USB 등으로 전달하여 오프라인 설치

## 1단계: 인터넷 환경에서 준비

### 1-1. Python 패키지 다운로드

```bash
# 오프라인 설치용 wheel 파일 다운로드
mkdir -p offline_packages

pip download airgap-kor-search \
    --dest ./offline_packages/ \
    --platform manylinux2014_x86_64 \
    --python-version 3.11 \
    --only-binary=:all:
```

> Windows 환경이라면 `--platform` 옵션을 `win_amd64`로 변경하세요.

소스에서 직접 빌드하는 경우:

```bash
# 프로젝트 클론 후 빌드
git clone https://github.com/Dev-2A/airgap-kor-search.git
cd airgap-kor-search
pip install build
python -m build

# 의존성 패키지도 다운로드
pip download -r &1 | grep -oP '[\w-]+==[.\d]+') \
    --dest ./offline_packages/
```

### 1-2. 임베딩 모델 준비

자세한 내용은 [모델 준비 가이드](model-preparation.md)를 참고하세요.

```bash
# ONNX 변환
pip install optimum[onnxruntime] transformers torch

optimum-cli export onnx \
    --model BAAI/bge-m3 \
    --task feature-extraction \
    ./bge-m3-onnx/

# 필요한 파일만 추출
mkdir -p transfer/model
cp bge-m3-onnx/model.onnx transfer/model/
cp bge-m3-onnx/tokenizer.json transfer/model/
```

### 1-3. 전달 패키지 구성

```bash
mkdir -p transfer

# 구조
transfer/
├── offline_packages/       # pip wheel 파일들
│   ├── airgap_kor_search-0.1.0-py3-none-any.whl
│   ├── faiss_cpu-1.7.4-*.whl
│   ├── onnxruntime-1.16.0-*.whl
│   └── ... (기타 의존성)
├── model/
│   ├── model.onnx          # ~2.2GB
│   └── tokenizer.json      # ~14MB
└── install.sh              # 설치 스크립트
```

### 1-4. 설치 스크립트 작성

```bash
# 파일: transfer/install.sh
#!/bin/bash
set -e

echo "=== airgap-kor-search 오프라인 설치 ==="

# 1. 패키지 설치
echo "[1/3] Python 패키지 설치 중..."
pip install --no-index --find-links=./offline_packages/ airgap-kor-search

# 2. 초기화
echo "[2/3] 설정 초기화..."
airgap-kor-search init

# 3. 모델 복사
echo"[3/3] 임베딩 모델 복사 중..."
cp -r ./model/* ./airgap_data/model/

echo ""
echo "=== 설치 완료! ==="
echo ""
echo "사용법:"
echo "  airgap-kor-search index ./문서폴더/"
echo "  airgap-kor-search search \"검색어\""
echo "  airgap-kor-search serve"
```

Windows용:

```bash
@echo off
REM 파일: transfer\install.bat

echo === airgap-kor-search 오프라인 설치 ===

echo [1/3] Python 패키지 설치 중...
pip install --no-index --find-links=.\offline_packages\ airgap-kor-search

echo [2/3] 설정 초기화...
airgap-kor-search init

3cho [3/3] 임베딩 모델 복사 중...
xcopy /E /I .\model\* .\airgap_data\model\

echo.
echo === 설치 완료! ===
echo.
echo 사용법:
echo   airgap-kor-search index .\문서폴더\
echo   airgap-kor-search search "검색어"
echo   airgap-kor-search serve
```

## 2단계: 에어갭 환경에서 설치

### 2-1. 전달

USB, CD, 또는 내부 네트워크를 통해 `transfer\` 디렉토리를 에어갭 환경으로 복사합니다.

### 2-2. 설치

```bash
cd transfer

# Linux/Mac
chmod +x install.sh
./install.sh

# Windows
install.bat
```

### 2-3. 확인

```bash
# 버전 확인
airgap-kor-search --version

# 초기화 확인
airgap-kor-search stats
```

## 시스템 요구사항

| 항목 | 최소 | 권장 |
| --- | --- | --- |
| Python | 3.9 | 3.11+ |
| RAM | 4GB | 8GB+ |
| 디스크 (모델 포함) | 3GB | 5GB+ |
| CPU | 2코어 | 4코어+ |
| GPU | 불필요 | 불필요 |

## 트러블슈팅

### Q: `pip install` 시 의존성 오류

모든 wheel 파일이 대상 환경의 플랫폼/Python 버전과 일치하는지 확인하세요.  
`pip download` 시 `--platform`과 `--python-version`을 정확히 지정해야 합니다.

### Q: 모델 로드 시 메모리 부족

양자화된 모델을 사용하세요. [모델 준비 가이드](model-preparation.md)의 경량화 섹션을 참고하세요.

### Q: 검색 속도가 느림

- 청크 수가 매우 많으면 FAISS IVF 인덱스로 전환을 고려하세요.
- `config.json`에서 `batch_size`를 줄이면 메모리 사용량이 감소합니다.
