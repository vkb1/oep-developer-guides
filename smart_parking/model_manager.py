"""Model download and management utilities using HuggingFace Hub."""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from smart_parking.config import (
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
    """Download the YOLOv11 license plate detection model from HuggingFace.

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

    logger.info("Downloading YOLO model from %s ...", repo_id)
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
    """Download the PP-OCRv4 server recognition model from HuggingFace.

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

    logger.info("Downloading PP-OCR model from %s ...", repo_id)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
    )
    marker.touch()
    logger.info("PP-OCR model saved to %s", output_dir)
    return output_dir


def download_all_models(
    yolo_repo: str = YOLO_LP_DETECTION_REPO,
    ppocr_repo: str = PPOCR_REC_REPO,
) -> tuple[Path, Path]:
    """Download all required models.

    Args:
        yolo_repo: HuggingFace repository ID for YOLO model.
        ppocr_repo: HuggingFace repository ID for PP-OCR model.

    Returns:
        Tuple of (yolo_model_path, ppocr_model_dir).
    """
    yolo_path = download_yolo_model(repo_id=yolo_repo)
    ppocr_dir = download_ppocr_model(repo_id=ppocr_repo)
    return yolo_path, ppocr_dir
