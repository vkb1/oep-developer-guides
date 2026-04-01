"""Model downloading utilities for HuggingFace Hub models."""

import logging
import os
import urllib.request

from huggingface_hub import snapshot_download

from . import config

logger = logging.getLogger(__name__)


def download_detection_model(models_dir: str = config.DEFAULT_MODELS_DIR) -> str:
    """Download the YOLOv11 license plate detection model from HuggingFace.

    Args:
        models_dir: Base directory for storing downloaded models.

    Returns:
        Path to the downloaded model directory.
    """
    local_dir = os.path.join(models_dir, "detection")
    model_path = os.path.join(local_dir, config.DETECTION_MODEL_FILENAME)

    if os.path.exists(model_path):
        logger.info("Detection model already downloaded: %s", model_path)
        return local_dir

    logger.info("Downloading detection model from %s ...", config.DETECTION_MODEL_REPO)
    snapshot_download(
        repo_id=config.DETECTION_MODEL_REPO,
        local_dir=local_dir,
    )
    logger.info("Detection model downloaded to %s", local_dir)
    return local_dir


def download_ocr_model(models_dir: str = config.DEFAULT_MODELS_DIR) -> str:
    """Download the PP-OCRv4 recognition model from HuggingFace.

    Args:
        models_dir: Base directory for storing downloaded models.

    Returns:
        Path to the downloaded model directory.
    """
    local_dir = os.path.join(models_dir, "ocr")
    model_path = os.path.join(local_dir, config.OCR_MODEL_PDMODEL)

    if os.path.exists(model_path):
        logger.info("OCR model already downloaded: %s", model_path)
        return local_dir

    logger.info("Downloading OCR model from %s ...", config.OCR_MODEL_REPO)
    snapshot_download(
        repo_id=config.OCR_MODEL_REPO,
        local_dir=local_dir,
    )
    logger.info("OCR model downloaded to %s", local_dir)
    return local_dir


def download_ocr_dictionary(models_dir: str = config.DEFAULT_MODELS_DIR) -> str:
    """Download the PP-OCR character dictionary.

    Args:
        models_dir: Base directory for storing downloaded models.

    Returns:
        Path to the downloaded dictionary file.
    """
    dict_dir = os.path.join(models_dir, "ocr")
    dict_path = os.path.join(dict_dir, config.OCR_DICT_FILENAME)

    if os.path.exists(dict_path):
        logger.info("OCR dictionary already downloaded: %s", dict_path)
        return dict_path

    os.makedirs(dict_dir, exist_ok=True)
    logger.info("Downloading OCR dictionary ...")

    # Validate URL scheme before downloading
    url = config.OCR_DICT_URL
    if not url.startswith("https://"):
        raise ValueError(f"OCR dictionary URL must use HTTPS: {url}")
    urllib.request.urlretrieve(url, dict_path)  # noqa: S310
    logger.info("OCR dictionary downloaded to %s", dict_path)
    return dict_path


def ensure_models(models_dir: str = config.DEFAULT_MODELS_DIR) -> tuple[str, str, str]:
    """Download all required models and return their paths.

    Args:
        models_dir: Base directory for storing downloaded models.

    Returns:
        Tuple of (detection_dir, ocr_dir, dict_path).
    """
    os.makedirs(models_dir, exist_ok=True)
    det_dir = download_detection_model(models_dir)
    ocr_dir = download_ocr_model(models_dir)
    dict_path = download_ocr_dictionary(models_dir)
    return det_dir, ocr_dir, dict_path
