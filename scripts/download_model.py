"""임베딩 모델 다운로드 & ONNX 변환 스크립트

인터넷이 되는 환경에서 실행하여 모델 파일을 준비합니다.

사용법:
    pip install optimum[onnxruntime] transformers torch
    python scripts/download_model.py
    python scripts/download_model.py --model intfloat/multilingual-e5-small --dim 384
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SUPPORTED_MODELS = {
    "BAAI/bge-m3": {"dim": 1024, "pooling": "cls"},
    "intfloat/multilingual-e5-large-instruct": {"dim": 1024, "pooling": "mean"},
    "intfloat/multilingual-e5-small": {"dim": 384, "pooling": "mean"},
}


def download_and_convert(
    model_name: str,
    output_dir: Path,
    quantize: bool = False,
) -> None:
    """모델을 다운로드하고 ONNX로 변환합니다."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
    except ImportError:
        print("❌ 필요한 패키지를 설치해주세요:")
        print("   pip install optimum[onnxruntime] transformers torch")
        sys.exit(1)

    print(f"📥 모델 다운로드 중: {model_name}")
    print(f"   (처음 실행 시 시간이 걸릴 수 있습니다)")
    print()

    # ONNX 변환 + 다운로드
    tmp_dir = output_dir / "_tmp_onnx"
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name, export=True
    )
    model.save_pretrained(str(tmp_dir))

    # tokenizer도 저장
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(tmp_dir))

    # 필요한 파일만 추출
    output_dir.mkdir(parents=True, exist_ok=True)

    # ONNX 모델 파일
    onnx_file = tmp_dir / "model.onnx"
    if not onnx_file.exists():
        # 일부 모델은 다른 이름일 수 있음
        onnx_files = list(tmp_dir.glob("*.onnx"))
        if onnx_files:
            onnx_file = onnx_files[0]
        else:
            print("❌ ONNX 파일을 찾을 수 없습니다.")
            sys.exit(1)

    shutil.copy2(onnx_file, output_dir / "model.onnx")
    print(f"✅ model.onnx ({_file_size(output_dir / 'model.onnx')})")

    # 토크나이저
    tokenizer_file = tmp_dir / "tokenizer.json"
    if tokenizer_file.exists():
        shutil.copy2(tokenizer_file, output_dir / "tokenizer.json")
        print(f"✅ tokenizer.json ({_file_size(output_dir / 'tokenizer.json')})")
    else:
        print("⚠️ tokenizer.json을 찾을 수 없습니다. 수동으로 복사해주세요.")

    # 양자화 (선택)
    if quantize:
        print("\n⚡ 양자화 진행 중...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(str(tmp_dir))
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

            quant_dir = output_dir / "_quantized"
            quantizer.quantize(
                save_dir=str(quant_dir), quantization_config=qconfig
            )

            quant_onnx = quant_dir / "model_quantized.onnx"
            if quant_onnx.exists():
                shutil.copy2(quant_onnx, output_dir / "model_quantized.onnx")
                print(
                    f"✅ model_quantized.onnx ({_file_size(output_dir / 'model_quantized.onnx')})"
                )

            shutil.rmtree(quant_dir, ignore_errors=True)
        except Exception as e:
            print(f"⚠️ 양자화 실패: {e}")

    # 임시 디렉토리 정리
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 모델 정보 파일
    info = SUPPORTED_MODELS.get(model_name, {})
    info_text = (
        f"model: {model_name}\n"
        f"dim: {info.get('dim', 'unknown')}\n"
        f"pooling: {info.get('pooling', 'unknown')}\n"
    )
    (output_dir / "model_info.txt").write_text(info_text)

    print(f"\n🎉 완료! 모델 파일 위치: {output_dir}")
    print(f"\n다음 단계:")
    print(f"  에어갭 환경으로 {output_dir} 디렉토리를 복사하세요.")


def _file_size(path: Path) -> str:
    """파일 크기를 사람이 읽기 쉬운 형식으로 반환"""
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def main():
    parser = argparse.ArgumentParser(
        description="임베딩 모델 다운로드 & ONNX 변환"
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help=f"모델 이름 (기본: BAAI/bge-m3). 지원: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--output",
        default="./airgap_data/model",
        help="출력 디렉토리 (기본: ./airgap_data/model)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="양자화 버전도 함께 생성",
    )

    args = parser.parse_args()
    download_and_convert(
        model_name=args.model,
        output_dir=Path(args.output),
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()