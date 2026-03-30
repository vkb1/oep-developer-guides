"""Download HuggingFace models for the Smart Parking application.

Downloads the following models:
  - YOLOv11 License Plate Detection from
    https://huggingface.co/morsetechlab/yolov11-license-plate-detection
  - PP-OCRv4 Server Recognition from
    https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec

Usage:
    python download_models.py
    python download_models.py --models-dir ./my_models
    python download_models.py --yolo-only
    python download_models.py --ppocr-only
"""

import argparse
import logging
import sys
from pathlib import Path

# --- Default Configuration ---
YOLO_REPO = "morsetechlab/yolov11-license-plate-detection"
YOLO_FILENAME = "yolov11-license-plate-detection.pt"

PPOCR_REPO = "PaddlePaddle/PP-OCRv4_server_rec"

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"

logger = logging.getLogger(__name__)


def download_yolo_model(
    repo_id: str = YOLO_REPO,
    filename: str = YOLO_FILENAME,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Path:
    """Download the YOLOv11 license plate detection model.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Model filename to download.
        models_dir: Base directory for storing models.

    Returns:
        Path to the downloaded model file.
    """
    output_dir = models_dir / "yolo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

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
        local_dir=str(output_dir),
    )
    downloaded = Path(downloaded_path)

    if downloaded != output_path and downloaded.exists():
        downloaded.rename(output_path)

    logger.info("YOLO model saved to %s", output_path)
    return output_path


def download_ppocr_model(
    repo_id: str = PPOCR_REPO,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Path:
    """Download the PP-OCRv4 server recognition model.

    Args:
        repo_id: HuggingFace repository ID.
        models_dir: Base directory for storing models.

    Returns:
        Path to the downloaded model directory.
    """
    output_dir = models_dir / "ppocr"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help=f"Directory to store downloaded models (default: {DEFAULT_MODELS_DIR}).",
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

    models_dir = Path(args.models_dir)
    download_yolo = not args.ppocr_only
    download_ppocr = not args.yolo_only

    try:
        if download_yolo:
            yolo_path = download_yolo_model(models_dir=models_dir)
            logger.info("YOLO model ready at: %s", yolo_path)

        if download_ppocr:
            ppocr_dir = download_ppocr_model(models_dir=models_dir)
            logger.info("PP-OCR model ready at: %s", ppocr_dir)

        logger.info("All requested models downloaded successfully.")
        return 0

    except Exception:
        logger.exception("Failed to download models")
        return 1


if __name__ == "__main__":
    sys.exit(main())
