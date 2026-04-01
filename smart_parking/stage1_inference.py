"""Stage 1: License plate detection and OCR using native HuggingFace models.

Downloads YOLOv11 for license plate detection and PP-OCRv4 for text
recognition, runs inference on an input image, and produces a side-by-side
comparison of the input, detection, and OCR results with timing metrics.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from smart_parking.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    OUTPUT_DIR,
    ensure_directories,
)
from smart_parking.model_manager import download_all_models
from smart_parking.visualization import create_comparison_figure, draw_detections

logger = logging.getLogger(__name__)


def _load_yolo_model(model_path: Path) -> YOLO:
    """Load the YOLOv11 model for license plate detection.

    Args:
        model_path: Path to the YOLO .pt model file.

    Returns:
        Loaded YOLO model instance.
    """
    logger.info("Loading YOLO model from %s", model_path)
    model = YOLO(str(model_path))
    return model


def _detect_license_plates(
    model: YOLO,
    image: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[list[dict], float]:
    """Run YOLO detection on an image to find license plates.

    Args:
        model: Loaded YOLO model.
        image: Input image (BGR format).
        confidence: Confidence threshold for detections.
        iou: IoU threshold for NMS.

    Returns:
        Tuple of (list of detection dicts, inference time in ms).
    """
    start = time.perf_counter()
    results = model.predict(image, conf=confidence, iou=iou, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = result.names.get(cls, str(cls))
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "label": label,
                    "class_id": cls,
                })

    logger.info("Detected %d license plate(s) in %.1f ms", len(detections), elapsed_ms)
    return detections, elapsed_ms


def _crop_plates(
    image: np.ndarray,
    detections: list[dict],
    padding: int = 5,
) -> list[np.ndarray]:
    """Crop detected license plate regions from the image.

    Args:
        image: Input image (BGR format).
        detections: List of detection dicts with 'bbox' key.
        padding: Pixel padding around the crop.

    Returns:
        List of cropped plate images.
    """
    h, w = image.shape[:2]
    crops = []
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops


def _recognize_plate_text(plate_crops: list[np.ndarray]) -> tuple[list[str], float]:
    """Recognize text from cropped license plate images using PaddleOCR.

    Args:
        plate_crops: List of cropped plate images (BGR format).

    Returns:
        Tuple of (list of recognized text strings, inference time in ms).
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    texts = []
    start = time.perf_counter()
    for crop in plate_crops:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = ocr.ocr(crop_rgb, cls=True)

        plate_text = ""
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    plate_text += text + " "
        texts.append(plate_text.strip() if plate_text.strip() else "[unreadable]")
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info("Recognized %d plate text(s) in %.1f ms", len(texts), elapsed_ms)
    return texts, elapsed_ms


def run_stage1(
    input_image: str,
    yolo_repo: str | None = None,
    ppocr_repo: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    output_dir: str | None = None,
) -> dict:
    """Execute Stage 1: Native model inference pipeline.

    Downloads models if needed, runs YOLO detection and PaddleOCR recognition
    on the input image, and produces a comparison visualization.

    Args:
        input_image: Path to the input car image.
        yolo_repo: Optional custom HuggingFace repo for YOLO model.
        ppocr_repo: Optional custom HuggingFace repo for PP-OCR model.
        confidence: Detection confidence threshold.
        output_dir: Optional output directory for results.

    Returns:
        Dict with keys: 'detections', 'texts', 'inference_times',
        'output_image', 'yolo_model_path', 'ppocr_model_dir'.
    """
    ensure_directories()
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download models
    kwargs = {}
    if yolo_repo:
        kwargs["yolo_repo"] = yolo_repo
    if ppocr_repo:
        kwargs["ppocr_repo"] = ppocr_repo
    yolo_path, ppocr_dir = download_all_models(**kwargs)

    # Load input image
    image_path = Path(input_image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    logger.info("Processing image: %s (%dx%d)", image_path.name, image.shape[1], image.shape[0])

    total_start = time.perf_counter()

    # Detection
    model = _load_yolo_model(yolo_path)
    detections, det_time = _detect_license_plates(model, image, confidence=confidence)

    # Draw detections
    annotated = draw_detections(image, detections)

    # Crop and OCR
    crops = _crop_plates(image, detections)
    texts, ocr_time = _recognize_plate_text(crops)

    total_ms = (time.perf_counter() - total_start) * 1000

    inference_times = {
        "detection_ms": det_time,
        "ocr_ms": ocr_time,
        "total_ms": total_ms,
    }

    # Save comparison figure
    output_path = create_comparison_figure(
        original=image,
        detection=annotated,
        ocr_texts=texts,
        inference_times=inference_times,
        title="Stage 1: Native Model Inference",
        output_path=out_dir / "comparison.png",
    )

    # Save annotated image
    annotated_path = out_dir / "annotated.png"
    cv2.imwrite(str(annotated_path), annotated)

    # Save plate crops
    for i, crop in enumerate(crops):
        crop_path = out_dir / f"plate_{i}.png"
        cv2.imwrite(str(crop_path), crop)

    logger.info(
        "Stage 1 complete - Detection: %.1f ms, OCR: %.1f ms, Total: %.1f ms",
        det_time,
        ocr_time,
        total_ms,
    )

    return {
        "detections": detections,
        "texts": texts,
        "inference_times": inference_times,
        "output_image": str(output_path),
        "yolo_model_path": str(yolo_path),
        "ppocr_model_dir": str(ppocr_dir),
    }
