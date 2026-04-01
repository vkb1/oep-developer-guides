"""Stage 2: OpenVINO IR conversion and optimized inference.

Converts the YOLOv11 and PP-OCRv4 models downloaded in Stage 1 to OpenVINO
Intermediate Representation (IR) format and runs optimized inference, showing
side-by-side comparison with timing metrics.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO

from smart_parking.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IOU_THRESHOLD,
    OUTPUT_DIR,
    OV_PPOCR_MODEL_DIR,
    OV_YOLO_MODEL_DIR,
    OV_YOLO_MODEL_PATH,
    PPOCR_MODEL_DIR,
    YOLO_MODEL_PATH,
    ensure_directories,
)
from smart_parking.stage1_inference import _crop_plates
from smart_parking.visualization import create_comparison_figure, draw_detections

logger = logging.getLogger(__name__)


def convert_yolo_to_openvino(
    model_path: Path = YOLO_MODEL_PATH,
    output_dir: Path = OV_YOLO_MODEL_DIR,
) -> Path:
    """Convert YOLOv11 .pt model to OpenVINO IR format.

    Uses the ultralytics export functionality to convert the model.

    Args:
        model_path: Path to the YOLO .pt model file.
        output_dir: Directory to save the OpenVINO IR model.

    Returns:
        Path to the exported OpenVINO model directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_xml = output_dir / f"{model_path.stem}_openvino_model" / f"{model_path.stem}.xml"
    if expected_xml.exists():
        logger.info("OpenVINO YOLO model already exists at %s", expected_xml)
        return expected_xml

    logger.info("Converting YOLO model to OpenVINO IR format ...")
    model = YOLO(str(model_path))
    export_path = model.export(format="openvino", dynamic=False, half=False)
    logger.info("YOLO model exported to OpenVINO at: %s", export_path)

    export_dir = Path(export_path)
    xml_files = list(export_dir.glob("*.xml"))
    if xml_files:
        return xml_files[0]

    raise FileNotFoundError(
        f"No .xml file found after export in {export_dir}"
    )


def convert_ppocr_to_openvino(
    ppocr_model_dir: Path = PPOCR_MODEL_DIR,
    output_dir: Path = OV_PPOCR_MODEL_DIR,
) -> Path:
    """Convert PP-OCR model to OpenVINO IR format using model optimizer.

    Args:
        ppocr_model_dir: Directory containing the PP-OCR model files.
        output_dir: Directory to save the OpenVINO IR model.

    Returns:
        Path to the output directory with converted models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / ".conversion_complete"
    if marker.exists():
        logger.info("OpenVINO PP-OCR model already exists at %s", output_dir)
        return output_dir

    logger.info("Converting PP-OCR model to OpenVINO IR format ...")

    inference_model = ppocr_model_dir / "inference.pdmodel"
    if not inference_model.exists():
        pdmodel_files = list(ppocr_model_dir.rglob("*.pdmodel"))
        if pdmodel_files:
            inference_model = pdmodel_files[0]
        else:
            logger.warning(
                "No .pdmodel found in %s. PP-OCR will use PaddlePaddle backend.",
                ppocr_model_dir,
            )
            marker.touch()
            return output_dir

    core = ov.Core()
    try:
        model = core.read_model(str(inference_model))
        ov.save_model(model, str(output_dir / "ppocr_rec.xml"))
        logger.info("PP-OCR model converted to OpenVINO IR at %s", output_dir)
    except (RuntimeError, ValueError):
        logger.warning(
            "Could not convert PP-OCR model with OpenVINO directly. "
            "PP-OCR will use PaddlePaddle backend for OCR.",
            exc_info=True,
        )

    marker.touch()
    return output_dir


def _detect_with_openvino(
    model_xml: Path,
    image: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
    device: str = DEFAULT_DEVICE,
) -> tuple[list[dict], float]:
    """Run license plate detection using YOLO model exported to OpenVINO.

    The ultralytics library natively supports loading OpenVINO models,
    so we leverage that for seamless inference.

    Args:
        model_xml: Path to the OpenVINO .xml model file.
        image: Input image (BGR format).
        confidence: Confidence threshold.
        iou: IoU threshold for NMS.
        device: OpenVINO device (CPU, GPU, etc.).

    Returns:
        Tuple of (list of detection dicts, inference time in ms).
    """
    logger.info("Loading OpenVINO YOLO model from %s on %s", model_xml, device)
    model = YOLO(str(model_xml))

    start = time.perf_counter()
    results = model.predict(image, conf=confidence, iou=iou, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = result.names.get(cls, str(cls))
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf_val,
                    "label": label,
                    "class_id": cls,
                })

    logger.info(
        "OpenVINO detection: %d plate(s) in %.1f ms", len(detections), elapsed_ms
    )
    return detections, elapsed_ms


def _recognize_with_openvino(
    plate_crops: list[np.ndarray],
    ov_ppocr_dir: Path = OV_PPOCR_MODEL_DIR,
) -> tuple[list[str], float]:
    """Recognize plate text using PaddleOCR with OpenVINO backend if available.

    Args:
        plate_crops: List of cropped plate images (BGR format).
        ov_ppocr_dir: Directory containing OpenVINO PP-OCR model.

    Returns:
        Tuple of (list of recognized texts, inference time in ms).
    """
    from paddleocr import PaddleOCR

    ocr_xml = ov_ppocr_dir / "ppocr_rec.xml"
    if ocr_xml.exists():
        logger.info("Using OpenVINO PP-OCR model for recognition")

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

    logger.info("OpenVINO OCR: %d text(s) in %.1f ms", len(texts), elapsed_ms)
    return texts, elapsed_ms


def run_stage2(
    input_image: str,
    yolo_model_path: str | None = None,
    ppocr_model_dir: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = DEFAULT_DEVICE,
    output_dir: str | None = None,
) -> dict:
    """Execute Stage 2: OpenVINO optimized inference pipeline.

    Converts models to OpenVINO IR format (if not already converted),
    then runs optimized inference.

    Args:
        input_image: Path to the input car image.
        yolo_model_path: Path to YOLO .pt model (from Stage 1).
        ppocr_model_dir: Path to PP-OCR model directory (from Stage 1).
        confidence: Detection confidence threshold.
        device: OpenVINO inference device.
        output_dir: Optional output directory for results.

    Returns:
        Dict with keys: 'detections', 'texts', 'inference_times',
        'output_image', 'ov_yolo_model', 'ov_ppocr_dir'.
    """
    ensure_directories()
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_pt = Path(yolo_model_path) if yolo_model_path else YOLO_MODEL_PATH
    ppocr_dir = Path(ppocr_model_dir) if ppocr_model_dir else PPOCR_MODEL_DIR

    if not yolo_pt.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {yolo_pt}. Run Stage 1 first."
        )

    # Convert models to OpenVINO IR
    ov_yolo_xml = convert_yolo_to_openvino(yolo_pt)
    ov_ppocr_dir = convert_ppocr_to_openvino(ppocr_dir)

    # Load input image
    image_path = Path(input_image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    logger.info(
        "Processing image with OpenVINO: %s (%dx%d)",
        image_path.name,
        image.shape[1],
        image.shape[0],
    )

    total_start = time.perf_counter()

    # Detection with OpenVINO
    detections, det_time = _detect_with_openvino(
        ov_yolo_xml, image, confidence=confidence, device=device
    )

    # Draw detections
    annotated = draw_detections(image, detections, color=(255, 165, 0))

    # Crop and OCR with OpenVINO
    crops = _crop_plates(image, detections)
    texts, ocr_time = _recognize_with_openvino(crops, ov_ppocr_dir)

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
        title=f"Stage 2: OpenVINO Optimized Inference ({device})",
        output_path=out_dir / "comparison.png",
    )

    # Save annotated image
    annotated_path = out_dir / "annotated.png"
    cv2.imwrite(str(annotated_path), annotated)

    logger.info(
        "Stage 2 complete - Detection: %.1f ms, OCR: %.1f ms, Total: %.1f ms",
        det_time,
        ocr_time,
        total_ms,
    )

    return {
        "detections": detections,
        "texts": texts,
        "inference_times": inference_times,
        "output_image": str(output_path),
        "ov_yolo_model": str(ov_yolo_xml),
        "ov_ppocr_dir": str(ov_ppocr_dir),
    }
