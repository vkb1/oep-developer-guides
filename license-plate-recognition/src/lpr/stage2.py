"""Stage 2: License plate detection and OCR using OpenVINO IR optimized models.

Converts the downloaded models to OpenVINO IR format and runs inference
using the optimized representations for improved performance.
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
from .stage1 import _ctc_decode, _load_ocr_dictionary, _preprocess_ocr_image
from .visualization import create_stage_comparison

logger = logging.getLogger(__name__)

# OpenVINO IR file names
_DET_IR_NAME = "detection"
_OCR_IR_NAME = "ocr"


def convert_models(models_dir: str = config.DEFAULT_MODELS_DIR) -> tuple[str, str]:
    """Convert downloaded models to OpenVINO IR format.

    Args:
        models_dir: Directory containing downloaded models.

    Returns:
        Tuple of (detection_ir_path, ocr_ir_path) as paths to the IR .xml files.
    """
    det_dir, ocr_dir, _ = ensure_models(models_dir)

    # --- Convert YOLO detection model ---
    ov_det_dir = os.path.join(models_dir, config.OV_DETECTION_SUBDIR)
    det_ir_path = os.path.join(ov_det_dir, f"{_DET_IR_NAME}.xml")

    if not os.path.exists(det_ir_path):
        logger.info("Converting detection model to OpenVINO IR ...")
        os.makedirs(ov_det_dir, exist_ok=True)

        det_model_path = os.path.join(det_dir, config.DETECTION_MODEL_FILENAME)
        yolo_model = YOLO(det_model_path)
        export_path = yolo_model.export(format="openvino", half=False, dynamic=True)

        # Move exported files to the target directory
        for fname in os.listdir(export_path):
            src = os.path.join(export_path, fname)
            dst = os.path.join(ov_det_dir, fname)
            if not os.path.exists(dst):
                os.rename(src, dst)

        # Rename to standard names
        for fname in os.listdir(ov_det_dir):
            base, ext = os.path.splitext(fname)
            if ext in (".xml", ".bin") and base != _DET_IR_NAME:
                new_name = f"{_DET_IR_NAME}{ext}"
                os.rename(
                    os.path.join(ov_det_dir, fname),
                    os.path.join(ov_det_dir, new_name),
                )

        logger.info("Detection IR saved to %s", ov_det_dir)
    else:
        logger.info("Detection IR already exists: %s", det_ir_path)

    # --- Convert OCR model ---
    ov_ocr_dir = os.path.join(models_dir, config.OV_OCR_SUBDIR)
    ocr_ir_path = os.path.join(ov_ocr_dir, f"{_OCR_IR_NAME}.xml")

    if not os.path.exists(ocr_ir_path):
        logger.info("Converting OCR model (PaddlePaddle) to OpenVINO IR ...")
        os.makedirs(ov_ocr_dir, exist_ok=True)

        ocr_model_path = os.path.join(ocr_dir, config.OCR_MODEL_PDMODEL)
        core = ov.Core()
        ov_model = core.read_model(ocr_model_path)
        ov.save_model(ov_model, ocr_ir_path)

        logger.info("OCR IR saved to %s", ov_ocr_dir)
    else:
        logger.info("OCR IR already exists: %s", ocr_ir_path)

    return det_ir_path, ocr_ir_path


def _run_ov_detection(
    model_path: str,
    image: np.ndarray,
    conf: float = config.DETECTION_CONFIDENCE,
    device: str = config.DEFAULT_DEVICE,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, float]], float]:
    """Run license plate detection using OpenVINO IR model.

    Args:
        model_path: Path to the detection IR .xml file.
        image: Input image (BGR).
        conf: Confidence threshold for detections.
        device: OpenVINO inference device.

    Returns:
        Tuple of (annotated_image, list_of_boxes, inference_time_ms).
        Each box is (x1, y1, x2, y2, confidence).
    """
    core = ov.Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, device)

    input_layer = compiled.input(0)
    _, c, h, w = input_layer.shape

    # Preprocess: resize, normalize, transpose
    original_h, original_w = image.shape[:2]
    resized = cv2.resize(image, (w, h))
    blob = resized.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    start = time.perf_counter()
    output = compiled(blob)
    inference_ms = (time.perf_counter() - start) * 1000

    # Parse YOLO output
    predictions = output[compiled.output(0)]
    boxes = _parse_yolo_output(predictions, original_w, original_h, w, h, conf)

    # Draw boxes on image
    annotated = image.copy()
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Plate {score:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, boxes, inference_ms


def _parse_yolo_output(
    predictions: np.ndarray,
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
    conf_threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    """Parse YOLO model output into bounding boxes.

    Handles the standard YOLO output format (1, num_features, num_detections)
    where num_features = 4 (bbox) + num_classes.

    Args:
        predictions: Raw model output.
        orig_w: Original image width.
        orig_h: Original image height.
        input_w: Model input width.
        input_h: Model input height.
        conf_threshold: Confidence threshold.

    Returns:
        List of (x1, y1, x2, y2, confidence) tuples in original image coordinates.
    """
    output = np.squeeze(predictions)

    # YOLO outputs can be (num_features, num_detections) or (num_detections, num_features)
    if output.ndim == 2 and output.shape[0] < output.shape[1]:
        output = output.T

    boxes = []
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    for detection in output:
        if len(detection) < 5:
            continue

        # Format: cx, cy, w, h, class_scores...
        cx, cy, w, h = detection[:4]
        class_scores = detection[4:]
        confidence = float(np.max(class_scores))

        if confidence < conf_threshold:
            continue

        x1 = int((cx - w / 2) * scale_x)
        y1 = int((cy - h / 2) * scale_y)
        x2 = int((cx + w / 2) * scale_x)
        y2 = int((cy + h / 2) * scale_y)

        # Clamp to image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        boxes.append((x1, y1, x2, y2, confidence))

    # Apply non-maximum suppression
    if boxes:
        boxes = _nms(boxes, iou_threshold=0.5)

    return boxes


def _nms(
    boxes: list[tuple[int, int, int, int, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, int, int, float]]:
    """Apply non-maximum suppression to bounding boxes.

    Args:
        boxes: List of (x1, y1, x2, y2, confidence) tuples.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Filtered list of boxes.
    """
    if not boxes:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []

    while sorted_boxes:
        best = sorted_boxes.pop(0)
        keep.append(best)
        remaining = []
        for box in sorted_boxes:
            if _iou(best, box) < iou_threshold:
                remaining.append(box)
        sorted_boxes = remaining

    return keep


def _iou(
    box_a: tuple[int, int, int, int, float],
    box_b: tuple[int, int, int, int, float],
) -> float:
    """Compute Intersection over Union between two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def run(
    input_path: str,
    models_dir: str = config.DEFAULT_MODELS_DIR,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
    device: str = config.DEFAULT_DEVICE,
) -> dict:
    """Run Stage 2: license plate detection and OCR with OpenVINO IR models.

    Args:
        input_path: Path to input image.
        models_dir: Directory containing downloaded models.
        output_dir: Directory for saving output.
        device: OpenVINO inference device (default: CPU).

    Returns:
        Dictionary with detection results, OCR text, and timing information.
    """
    logger.info("=== Stage 2: OpenVINO IR Optimized Inference ===")

    # Convert models to IR format (if not already converted)
    det_ir_path, ocr_ir_path = convert_models(models_dir)

    # Ensure dictionary is available
    _, _, dict_path = ensure_models(models_dir)

    # Load input image
    input_image = cv2.imread(input_path)
    if input_image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    h, w = input_image.shape[:2]
    logger.info("Input image loaded: %s (%dx%d)", input_path, w, h)

    # --- Detection with OpenVINO ---
    detection_image, det_boxes, detection_time_ms = _run_ov_detection(
        det_ir_path, input_image, device=device
    )
    logger.info("Detection: found %d plates in %.1f ms", len(det_boxes), detection_time_ms)

    # --- OCR with OpenVINO ---
    characters = _load_ocr_dictionary(dict_path)
    core = ov.Core()
    ocr_model = core.read_model(ocr_ir_path)
    compiled_ocr = core.compile_model(ocr_model, device)

    all_texts = []
    total_ocr_time_ms = 0.0

    for i, (x1, y1, x2, y2, _conf) in enumerate(det_boxes):
        plate_crop = input_image[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

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
    output_path = os.path.join(output_dir, "stage2_comparison.png")
    create_stage_comparison(
        input_image=input_image,
        detection_image=detection_image,
        ocr_text=combined_text,
        detection_time_ms=detection_time_ms,
        ocr_time_ms=total_ocr_time_ms,
        output_path=output_path,
        stage_label="Stage 2 (OpenVINO IR)",
    )

    return {
        "stage": 2,
        "plates_detected": len(det_boxes),
        "ocr_texts": all_texts,
        "detection_time_ms": detection_time_ms,
        "ocr_time_ms": total_ocr_time_ms,
        "total_time_ms": detection_time_ms + total_ocr_time_ms,
        "output_path": output_path,
    }
