"""Smart Parking Application - CLI entry point.

A generic Python application for license plate detection and recognition
that supports three execution stages:

  Stage 1: Native inference using HuggingFace models (YOLO + PaddleOCR)
  Stage 2: OpenVINO optimized inference with IR model conversion
  Stage 3: DL Streamer pipeline for video stream processing

Usage:
    python -m smart_parking.main --stage 1 --input image.jpg
    python -m smart_parking.main --stage 2 --input image.jpg --device CPU
    python -m smart_parking.main --stage 3 --input video.mp4 --device CPU
"""

import argparse
import json
import logging
import sys


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="smart_parking",
        description=(
            "Smart Parking - License Plate Detection & Recognition\n\n"
            "A generic application that takes an input image or video,\n"
            "detects license plates using YOLO, extracts text via OCR,\n"
            "and optionally optimizes with OpenVINO or DL Streamer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help=(
            "Execution stage: "
            "1 = Native HuggingFace model inference, "
            "2 = OpenVINO optimized inference, "
            "3 = DL Streamer video pipeline"
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image (stages 1-2) or video (stage 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: output/stage<N>).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Inference device for OpenVINO (default: CPU).",
    )
    parser.add_argument(
        "--yolo-repo",
        type=str,
        default=None,
        help="Custom HuggingFace repo for YOLO model.",
    )
    parser.add_argument(
        "--ppocr-repo",
        type=str,
        default=None,
        help="Custom HuggingFace repo for PP-OCR model.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="Path to local YOLO model file (skip download).",
    )
    parser.add_argument(
        "--ppocr-model-dir",
        type=str,
        default=None,
        help="Path to local PP-OCR model directory (skip download).",
    )
    parser.add_argument(
        "--ov-yolo-model",
        type=str,
        default=None,
        help="Path to OpenVINO YOLO model XML (stage 2-3).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process in video (stage 3, 0 = all).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the Smart Parking application.

    Args:
        argv: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Smart Parking Application - Stage %d", args.stage)

    try:
        if args.stage == 1:
            from smart_parking.stage1_inference import run_stage1

            result = run_stage1(
                input_image=args.input,
                yolo_repo=args.yolo_repo,
                ppocr_repo=args.ppocr_repo,
                confidence=args.confidence,
                output_dir=args.output_dir,
            )

        elif args.stage == 2:
            from smart_parking.stage2_openvino import run_stage2

            result = run_stage2(
                input_image=args.input,
                yolo_model_path=args.yolo_model,
                ppocr_model_dir=args.ppocr_model_dir,
                confidence=args.confidence,
                device=args.device,
                output_dir=args.output_dir,
            )

        elif args.stage == 3:
            from smart_parking.stage3_dlstreamer import run_stage3

            result = run_stage3(
                input_video=args.input,
                ov_yolo_model=args.ov_yolo_model,
                device=args.device,
                confidence=args.confidence,
                max_frames=args.max_frames,
                output_dir=args.output_dir,
            )

        # Print results summary
        logger.info("=" * 60)
        logger.info("Results:")
        logger.info("=" * 60)

        # Use json for clean output, handling non-serializable types
        def _default_serializer(obj):
            return str(obj)

        print(json.dumps(result, indent=2, default=_default_serializer))

        if "inference_times" in result:
            times = result["inference_times"]
            logger.info("Detection: %.1f ms", times.get("detection_ms", 0))
            logger.info("OCR: %.1f ms", times.get("ocr_ms", 0))
            logger.info("Total: %.1f ms", times.get("total_ms", 0))

        if "avg_fps" in result:
            logger.info("Average FPS: %.1f", result["avg_fps"])

        if "output_image" in result:
            logger.info("Output saved to: %s", result["output_image"])

        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except Exception:
        logger.exception("Stage %d failed", args.stage)
        return 1


if __name__ == "__main__":
    sys.exit(main())
