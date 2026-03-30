"""Download HuggingFace models for the Smart Parking application.

Downloads the following models:
  - YOLOv11 License Plate Detection from
    https://huggingface.co/morsetechlab/yolov11-license-plate-detection
  - PP-OCRv4 Server Recognition from
    https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec

Optionally converts downloaded models to OpenVINO IR format when the
``--convert-openvino`` flag is provided.

Models are saved to the top-level ``models/`` directory as configured in
``smart_parking.config``.

Usage:
    python download_models.py
    python download_models.py --yolo-only
    python download_models.py --ppocr-only
    python download_models.py --convert-openvino
"""

import argparse
import logging
import sys
from pathlib import Path

from smart_parking.config import (
    MODELS_DIR,
    PPOCR_MODEL_DIR,
    PPOCR_REC_REPO,
    YOLO_LP_DETECTION_FILENAME,
    YOLO_LP_DETECTION_REPO,
    YOLO_MODEL_PATH,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def download_yolo_model(
    repo_id: str = YOLO_LP_DETECTION_REPO,
    filename: str = YOLO_LP_DETECTION_FILENAME,
    output_path: Path = YOLO_MODEL_PATH,
) -> Path:
    """Download the YOLOv11 license plate detection model.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Model filename to download.
        output_path: Local path to save the downloaded model.

    Returns:
        Path to the downloaded model file.
    """
    ensure_directories()

    if output_path.exists():
        logger.info("YOLO model already exists at %s", output_path)
        return output_path

    from huggingface_hub import hf_hub_download

    logger.info(
        "Downloading YOLO model from https://huggingface.co/%s ...", repo_id
    )
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_path.parent),
    )
    downloaded = Path(downloaded_path)

    if downloaded != output_path and downloaded.exists():
        downloaded.rename(output_path)

    logger.info("YOLO model saved to %s", output_path)
    return output_path


def download_ppocr_model(
    repo_id: str = PPOCR_REC_REPO,
    output_dir: Path = PPOCR_MODEL_DIR,
) -> Path:
    """Download the PP-OCRv4 server recognition model.

    Args:
        repo_id: HuggingFace repository ID.
        output_dir: Local directory to save the downloaded model.

    Returns:
        Path to the downloaded model directory.
    """
    ensure_directories()

    marker = output_dir / ".download_complete"
    if marker.exists():
        logger.info("PP-OCR model already exists at %s", output_dir)
        return output_dir

    from huggingface_hub import snapshot_download

    logger.info(
        "Downloading PP-OCR model from https://huggingface.co/%s ...", repo_id
    )
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
    )
    marker.touch()
    logger.info("PP-OCR model saved to %s", output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and download requested models.

    Args:
        argv: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for Smart Parking application."
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Download only the YOLO license plate detection model.",
    )
    parser.add_argument(
        "--ppocr-only",
        action="store_true",
        help="Download only the PP-OCR recognition model.",
    )
    parser.add_argument(
        "--convert-openvino",
        action="store_true",
        help="Convert downloaded models to OpenVINO IR format after downloading.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Models directory: %s", MODELS_DIR)

    do_yolo = not args.ppocr_only
    do_ppocr = not args.yolo_only

    try:
        if do_yolo:
            yolo_path = download_yolo_model()
            logger.info("YOLO model ready at: %s", yolo_path)

        if do_ppocr:
            ppocr_dir = download_ppocr_model()
            logger.info("PP-OCR model ready at: %s", ppocr_dir)

        logger.info("All requested models downloaded successfully.")

        if args.convert_openvino:
            try:
                from smart_parking.stage2_openvino import (
                    convert_ppocr_to_openvino,
                    convert_yolo_to_openvino,
                )
            except ImportError:
                logger.error(
                    "OpenVINO dependencies are not installed. "
                    "Install openvino and ultralytics to use --convert-openvino."
                )
                return 1

            logger.info("Converting models to OpenVINO IR format ...")

            if do_yolo:
                ov_yolo_xml = convert_yolo_to_openvino()
                if ov_yolo_xml and ov_yolo_xml.exists():
                    logger.info("OpenVINO YOLO model ready at: %s", ov_yolo_xml)
                else:
                    logger.error("YOLO OpenVINO IR conversion failed.")
                    return 1

            if do_ppocr:
                ov_ppocr_dir = convert_ppocr_to_openvino()
                if ov_ppocr_dir and ov_ppocr_dir.exists():
                    logger.info("OpenVINO PP-OCR model ready at: %s", ov_ppocr_dir)
                else:
                    logger.error("PP-OCR OpenVINO IR conversion failed.")
                    return 1

            logger.info("OpenVINO IR conversion complete.")

        return 0

    except Exception:
        logger.exception("Failed to download models")
        return 1


if __name__ == "__main__":
    sys.exit(main())
