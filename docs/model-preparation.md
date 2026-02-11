# 🔧 모델 준비 가이드

에어갭 환경에서 사용하기 위해 임베딩 모델을 ONNX 형식으로 변환하고  
필요한 파일을 준비하는 방법을 설명합니다.

## 지원 모델

| 모델 | Pooling | 차원 | 권장 용도 |
| --- | --- | --- | --- |
| BAAI/bge-m3 (기본) | CLS | 1024 | 최고 성능, 범용 |
| intfloat/multilingual-e5-large-instruct | Mean | 1024 | E5 계열 선호 시 |
| intfloat/multilingual-e5-small | Mean | 384 | 저사양 환경 |

## 준비 과정 (인터넷이 되는 환경에서)

### 1. 필요 패키지 설치

```bash
pip install optimum[onnxruntime] transformers torch
```

### 2. ONNX 변환

```bash
#BGE-M3 (기본)
optimum-cli export onnx \
    --model BAAI/bge-m3 \
    --task feature-extraction \
    ./bge-m3-onnx/
```

### 3. 필요한 파일만 추출

변환 후 디렉토리에서 아래 2개 파일만 필요합니다.

```text
model_dir/
├── model.onnx          # ONNX 모델 파일
└── tokenizer.json      # 토크나이저 파일
```

```bash
# 필요한 파일만 복사
mkdir -p airgap_data/model
cp bge-m3-onnx/model.onnx airgap_data/model/
cp bge-m3-onnx/tokenizer.json airgap_data/model/
```

### 4. 에어갭 환경으로 전달

USB 등으로 `airgap_data/model/` 디렉토리를 에어갭 환경으로 복사합니다.

## 경량화 (선택사항)

모델 크기를 줄이고 싶다면 ONNX 양자화를 적용할 수 있습니다.

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("./bge-m3-onnx/")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer.quantize(save_dir="./bge-m3-onnx-quantized/", quantization_config=qconfig)
```

이렇게 하면 모델 크기가 약 1/4로 줄어들며, 성능 저하는 미미합니다.

## 다른 모델 사용 시 설정

`config.json`에서 모델 설정을 변경합니다.

```json
{
  "model": {
    "model_dir": "./airgap_data/model",
    "embedding_dim": 384,
    "max_seq_length": 512
  }
}
```

Mean Pooling 모델(E5 등)을 사용할 경우,  
코드에서 `MeanPoolingEmbedder`를 사용하세요.
