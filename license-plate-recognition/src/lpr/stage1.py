"""Stage 1: License plate detection and OCR using natively loaded models.

Uses ultralytics for YOLO detection and OpenVINO Runtime to load the
PaddlePaddle OCR model directly (on-the-fly, without prior IR conversion).
"""

import logging
import os
import time

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO

from . import config
from .downloader import ensure_models
from .visualization import create_stage_comparison

logger = logging.getLogger(__name__)


def _load_ocr_dictionary(dict_path: str) -> list[str]:
    """Load the PP-OCR character dictionary.

    Args:
        dict_path: Path to the dictionary file.

    Returns:
        List of characters including the blank token at index 0.
    """
    with open(dict_path, encoding="utf-8") as f:
        characters = [line.strip("\n") for line in f.readlines()]
    # Index 0 is reserved for the CTC blank token; append a space for the last index
    characters = ["<blank>"] + characters + [" "]
    return characters


def _preprocess_ocr_image(image: np.ndarray) -> np.ndarray:
    """Preprocess a cropped plate image for PP-OCRv4 recognition.

    Resizes to the expected input height while preserving aspect ratio,
    pads to target width, and normalizes pixel values.

    Args:
        image: Cropped license plate image (BGR, HxWxC).

    Returns:
        Preprocessed image tensor (1, C, H, W) as float32.
    """
    target_h = config.OCR_INPUT_HEIGHT
    target_w = config.OCR_INPUT_WIDTH

    h, w = image.shape[:2]
    ratio = target_h / h
    resized_w = min(int(w * ratio), target_w)
    resized = cv2.resize(image, (resized_w, target_h))

    # Convert BGR to RGB, normalize to [-1, 1]
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    resized = (resized / 255.0 - 0.5) / 0.5

    # Pad to target width
    padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
    padded[:, :resized_w, :] = resized

    # Transpose to CHW and add batch dimension
    tensor = np.transpose(padded, (2, 0, 1))[np.newaxis, ...]
    return tensor


def _ctc_decode(predictions: np.ndarray, characters: list[str]) -> str:
    """Decode CTC output predictions to text.

    Args:
        predictions: Model output array of shape (1, seq_len, num_classes).
        characters: Character dictionary.

    Returns:
        Decoded text string.
    """
    # Take argmax along the class dimension
    indices = np.argmax(predictions[0], axis=1)

    # Remove consecutive duplicates and blanks (index 0)
    result_chars = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx:
            if idx < len(characters):
                result_chars.append(characters[idx])
        prev_idx = idx

    return "".join(result_chars).strip()


def run(
    input_path: str,
    models_dir: str = config.DEFAULT_MODELS_DIR,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run Stage 1: license plate detection and OCR with natively loaded models.

    Args:
        input_path: Path to input image.
        models_dir: Directory containing downloaded models.
        output_dir: Directory for saving output.

    Returns:
        Dictionary with detection results, OCR text, and timing information.
    """
    logger.info("=== Stage 1: Native Model Inference ===")

    # Ensure models are downloaded
    det_dir, ocr_dir, dict_path = ensure_models(models_dir)

    # Load input image
    input_image = cv2.imread(input_path)
    if input_image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    h, w = input_image.shape[:2]
    logger.info("Input image loaded: %s (%dx%d)", input_path, w, h)

    # --- Detection ---
    det_model_path = os.path.join(det_dir, config.DETECTION_MODEL_FILENAME)
    logger.info("Loading YOLO detection model: %s", det_model_path)
    yolo_model = YOLO(det_model_path)

    start_time = time.perf_counter()
    results = yolo_model(input_image, conf=config.DETECTION_CONFIDENCE, verbose=False)
    detection_time_ms = (time.perf_counter() - start_time) * 1000

    # Draw detection results
    detection_image = results[0].plot() if results else input_image.copy()
    boxes = results[0].boxes if results else []
    logger.info("Detection: found %d plates in %.1f ms", len(boxes), detection_time_ms)

    # --- OCR ---
    characters = _load_ocr_dictionary(dict_path)

    # Load PaddlePaddle model directly via OpenVINO (on-the-fly conversion)
    ocr_model_path = os.path.join(ocr_dir, config.OCR_MODEL_PDMODEL)
    logger.info("Loading OCR model (PaddlePaddle format): %s", ocr_model_path)
    core = ov.Core()
    ocr_model = core.read_model(ocr_model_path)
    compiled_ocr = core.compile_model(ocr_model, "CPU")

    all_texts = []
    total_ocr_time_ms = 0.0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(input_image.shape[1], x2), min(input_image.shape[0], y2)

        plate_crop = input_image[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        # Preprocess and run OCR
        ocr_input = _preprocess_ocr_image(plate_crop)

        start_time = time.perf_counter()
        ocr_output = compiled_ocr(ocr_input)
        ocr_time = (time.perf_counter() - start_time) * 1000
        total_ocr_time_ms += ocr_time

        predictions = ocr_output[compiled_ocr.output(0)]
        text = _ctc_decode(predictions, characters)
        all_texts.append(text)
        logger.info("  Plate %d: '%s' (%.1f ms)", i + 1, text, ocr_time)

    combined_text = "\n".join(all_texts) if all_texts else ""
    if not all_texts:
        logger.warning("No license plate text detected")

    # --- Visualization ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "stage1_comparison.png")
    create_stage_comparison(
        input_image=input_image,
        detection_image=detection_image,
        ocr_text=combined_text,
        detection_time_ms=detection_time_ms,
        ocr_time_ms=total_ocr_time_ms,
        output_path=output_path,
        stage_label="Stage 1 (Native Models)",
    )

    return {
        "stage": 1,
        "plates_detected": len(boxes),
        "ocr_texts": all_texts,
        "detection_time_ms": detection_time_ms,
        "ocr_time_ms": total_ocr_time_ms,
        "total_time_ms": detection_time_ms + total_ocr_time_ms,
        "output_path": output_path,
    }
