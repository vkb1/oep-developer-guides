"""CLI entry point for the License Plate Recognition application.

Usage:
    python -m lpr --stage 1 --input path/to/car.jpg
    python -m lpr --stage 2 --input path/to/car.jpg
    python -m lpr --stage 3 --input path/to/video.mp4 --device CPU
"""

import argparse
import json
import logging
import sys

from . import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lpr",
        description=(
            "License Plate Recognition — multi-stage inference pipeline using "
            "YOLO detection, PP-OCRv4 recognition, OpenVINO optimization, "
            "and DL Streamer video processing."
        ),
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help=(
            "Pipeline stage to run: "
            "1 = native model inference, "
            "2 = OpenVINO IR optimized inference, "
            "3 = DL Streamer video pipeline."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to input image (stages 1-2) or video/stream (stage 3). "
            "For stage 3, defaults to the sample parking video URL."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=config.DEFAULT_MODELS_DIR,
        help=f"Directory for model storage (default: {config.DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {config.DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEFAULT_DEVICE,
        choices=["CPU", "GPU", "AUTO"],
        help="Inference device for stages 2-3 (default: CPU).",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default=None,
        help=(
            "Stage 3 output mode: display, fps, json, display-and-json, file "
            "(default: json)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the LPR application."""
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.stage in (1, 2) and args.input is None:
        print("Error: --input is required for stages 1 and 2.", file=sys.stderr)
        sys.exit(1)

    if args.stage == 1:
        from . import stage1

        result = stage1.run(
            input_path=args.input,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
        )

    elif args.stage == 2:
        from . import stage2

        result = stage2.run(
            input_path=args.input,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
        )

    elif args.stage == 3:
        from . import stage3

        output_mode = args.output_mode or "json"
        result = stage3.run(
            input_path=args.input,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            device=args.device,
            output_mode=output_mode,
        )

    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
