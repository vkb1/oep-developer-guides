"""Convert HuggingFace models to OpenVINO IR format for optimized inference.

This script converts:
- YOLOv11 license plate detection model → OpenVINO IR (.xml/.bin)
- TrOCR text recognition model → OpenVINO IR (.xml/.bin)
"""

import argparse
from pathlib import Path

import openvino as ov
from ultralytics import YOLO

DEFAULT_YOLO_MODEL = "morsetechlab/yolov11-license-plate-detection"
DEFAULT_OUTPUT_DIR = Path("models_ov")


def convert_yolo_to_openvino(model_path: str, output_dir: Path) -> Path:
    """Convert YOLO model to OpenVINO IR format using ultralytics export.

    Args:
        model_path: HuggingFace model ID or local path to the YOLO model.
        output_dir: Directory to save the converted model.

    Returns:
        Path to the converted OpenVINO model XML file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nConverting YOLO model: {model_path}")
    print(f"Output directory: {output_dir}")

    # Load YOLO model
    model = YOLO(model_path)

    # Export to OpenVINO format
    export_path = model.export(format="openvino")
    print(f"YOLO model exported to: {export_path}")

    return Path(export_path)


def convert_trocr_to_openvino(model_path: str, output_dir: Path) -> Path:
    """Convert TrOCR model to OpenVINO IR format.

    The TrOCR model consists of an encoder-decoder architecture. This function
    converts the full model pipeline to OpenVINO IR format using optimum-intel
    or direct ONNX conversion with OpenVINO tools.

    Args:
        model_path: HuggingFace model ID or local path to the TrOCR model.
        output_dir: Directory to save the converted model.

    Returns:
        Path to the output directory containing the converted model files.
    """
    trocr_output = output_dir / "trocr"
    trocr_output.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting TrOCR model: {model_path}")
    print(f"Output directory: {trocr_output}")

    try:
        # Try using optimum-intel for direct conversion
        from optimum.intel import OVModelForVision2Seq

        ov_model = OVModelForVision2Seq.from_pretrained(
            model_path, export=True
        )
        ov_model.save_pretrained(str(trocr_output))
        print(f"TrOCR model converted with optimum-intel to: {trocr_output}")

    except ImportError:
        # Fallback: Manual ONNX export then convert to OpenVINO IR
        print("optimum-intel not available, using manual ONNX → OpenVINO conversion")
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.eval()

        # Export encoder to ONNX
        encoder_onnx_path = trocr_output / "encoder.onnx"
        dummy_input = torch.randn(1, 3, 384, 384)

        print("  Exporting encoder to ONNX...")
        encoder = model.encoder
        torch.onnx.export(
            encoder,
            dummy_input,
            str(encoder_onnx_path),
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
            },
            opset_version=14,
        )

        # Convert ONNX to OpenVINO IR
        print("  Converting encoder ONNX to OpenVINO IR...")
        ov_encoder = ov.convert_model(str(encoder_onnx_path))
        encoder_ir_path = trocr_output / "encoder.xml"
        ov.save_model(ov_encoder, str(encoder_ir_path))

        # Save processor for inference
        processor.save_pretrained(str(trocr_output))

        # Clean up ONNX file
        encoder_onnx_path.unlink(missing_ok=True)

        print(f"TrOCR encoder converted to: {encoder_ir_path}")

    print("TrOCR conversion complete!")
    return trocr_output


def main() -> None:
    """Main entry point for model conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to OpenVINO IR format"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=DEFAULT_YOLO_MODEL,
        help=f"YOLO model path or HuggingFace ID (default: {DEFAULT_YOLO_MODEL})",
    )
    parser.add_argument(
        "--trocr-model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="TrOCR model path or HuggingFace ID (default: microsoft/trocr-base-printed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for converted models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Convert only the YOLO model",
    )
    parser.add_argument(
        "--trocr-only",
        action="store_true",
        help="Convert only the TrOCR model",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    convert_yolo = not args.trocr_only
    convert_trocr = not args.yolo_only

    if convert_yolo:
        yolo_ir_path = convert_yolo_to_openvino(args.yolo_model, output_dir / "yolo")
        print(f"\nYOLO OpenVINO model: {yolo_ir_path}")

    if convert_trocr:
        trocr_ir_path = convert_trocr_to_openvino(args.trocr_model, output_dir)
        print(f"TrOCR OpenVINO model: {trocr_ir_path}")

    print("\nModel conversion complete!")
    print(f"All models saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
