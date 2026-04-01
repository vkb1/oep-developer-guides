"""V2 Sample App: License Plate Detection using OpenVINO Optimized Models.

This application uses OpenVINO IR optimized versions of:
- YOLOv11 for license plate detection
- TrOCR for text extraction from detected plates

It compares performance between the original HuggingFace models and the
OpenVINO optimized models, displaying side-by-side results with inference timing.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

# Use non-interactive backend when no display is available
matplotlib.use("Agg")

# Default paths
DEFAULT_YOLO_OV_MODEL = "models_ov/yolo"
DEFAULT_TROCR_OV_MODEL = "models_ov/trocr"
DEFAULT_YOLO_HF_MODEL = "morsetechlab/yolov11-license-plate-detection"
DEFAULT_TROCR_HF_MODEL = "microsoft/trocr-base-printed"
OUTPUT_DIR = Path("output")


def load_yolo_openvino(model_dir: str) -> YOLO:
    """Load the YOLO model from OpenVINO IR format.

    Args:
        model_dir: Path to directory containing the OpenVINO model files.

    Returns:
        Loaded YOLO model configured for OpenVINO inference.
    """
    model_dir = Path(model_dir)

    # Find the OpenVINO XML file
    xml_files = list(model_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No OpenVINO XML model found in {model_dir}. "
            "Run convert_models.py first."
        )

    model_path = xml_files[0]
    print(f"Loading YOLO OpenVINO model from: {model_path}")
    start_time = time.time()
    model = YOLO(str(model_path), task="detect")
    load_time = time.time() - start_time
    print(f"YOLO OpenVINO model loaded in {load_time:.2f}s")
    return model


def load_trocr_openvino(
    model_dir: str,
) -> tuple:
    """Load the TrOCR model from OpenVINO IR format.

    Attempts to load using optimum-intel OVModelForVision2Seq first,
    then falls back to using the original HuggingFace model with OpenVINO
    runtime if the optimum-intel format is not available.

    Args:
        model_dir: Path to directory containing the OpenVINO model files.

    Returns:
        Tuple of (processor, model, is_openvino) where is_openvino indicates
        whether the model is running with OpenVINO optimization.
    """
    model_dir = Path(model_dir)
    print(f"Loading TrOCR OpenVINO model from: {model_dir}")
    start_time = time.time()

    try:
        from optimum.intel import OVModelForVision2Seq

        processor = TrOCRProcessor.from_pretrained(str(model_dir))
        model = OVModelForVision2Seq.from_pretrained(str(model_dir))
        load_time = time.time() - start_time
        print(f"TrOCR OpenVINO model loaded (optimum-intel) in {load_time:.2f}s")
        return processor, model, True

    except (ImportError, Exception) as e:
        print(f"  optimum-intel loading failed: {e}")
        print("  Falling back to HuggingFace model with OpenVINO runtime...")

        processor = TrOCRProcessor.from_pretrained(DEFAULT_TROCR_HF_MODEL)
        model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_TROCR_HF_MODEL)
        load_time = time.time() - start_time
        print(f"TrOCR HuggingFace model loaded as fallback in {load_time:.2f}s")
        return processor, model, False


def detect_license_plates_ov(
    model: YOLO, image: np.ndarray, confidence: float = 0.25
) -> tuple[list[dict], float]:
    """Detect license plates using OpenVINO optimized YOLO model.

    Args:
        model: Loaded YOLO model (OpenVINO backend).
        image: Input image as numpy array (BGR format).
        confidence: Minimum confidence threshold for detections.

    Returns:
        Tuple of (list of detection dicts, inference time in seconds).
    """
    start_time = time.time()
    results = model(image, conf=confidence, verbose=False)
    inference_time = time.time() - start_time

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class": cls,
                    }
                )

    return detections, inference_time


def extract_text_ov(
    processor, model, plate_image: np.ndarray, is_openvino: bool
) -> tuple[str, float]:
    """Extract text from a license plate image using TrOCR (OpenVINO or HuggingFace).

    Args:
        processor: TrOCR processor.
        model: TrOCR model (OpenVINO or HuggingFace).
        plate_image: Cropped license plate image as numpy array (BGR format).
        is_openvino: Whether the model is running with OpenVINO optimization.

    Returns:
        Tuple of (extracted text string, inference time in seconds).
    """
    plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
    plate_pil = Image.fromarray(plate_rgb)

    start_time = time.time()
    pixel_values = processor(images=plate_pil, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    inference_time = time.time() - start_time

    return text.strip(), inference_time


def draw_detections(
    image: np.ndarray, detections: list[dict], plate_texts: list[str]
) -> np.ndarray:
    """Draw bounding boxes and labels on the image.

    Args:
        image: Input image as numpy array (BGR format).
        detections: List of detection dicts with 'bbox' and 'confidence' keys.
        plate_texts: List of extracted text strings corresponding to detections.

    Returns:
        Annotated image as numpy array.
    """
    annotated = image.copy()

    for detection, text in zip(detections, plate_texts):
        x1, y1, x2, y2 = detection["bbox"]
        conf = detection["confidence"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{text} ({conf:.2f})"
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return annotated


def run_v1_pipeline(
    image: np.ndarray, confidence: float
) -> tuple[list[dict], list[str], float, list[float]]:
    """Run the V1 (HuggingFace) pipeline for comparison.

    Args:
        image: Input image as numpy array (BGR format).
        confidence: Detection confidence threshold.

    Returns:
        Tuple of (detections, plate_texts, detection_time, ocr_times).
    """
    from v1_sample_app.v1_license_plate_detection import (
        detect_license_plates,
        extract_text_from_plate,
        load_trocr_model,
        load_yolo_model,
    )

    yolo_model = load_yolo_model(DEFAULT_YOLO_HF_MODEL)
    trocr_processor, trocr_model = load_trocr_model(DEFAULT_TROCR_HF_MODEL)

    detections, detection_time = detect_license_plates(yolo_model, image, confidence)

    plate_texts = []
    ocr_times = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        plate_crop = image[y1:y2, x1:x2]
        if plate_crop.size == 0:
            plate_texts.append("N/A")
            ocr_times.append(0.0)
            continue
        text, ocr_time = extract_text_from_plate(trocr_processor, trocr_model, plate_crop)
        plate_texts.append(text)
        ocr_times.append(ocr_time)

    return detections, plate_texts, detection_time, ocr_times


def create_comparison_figure(
    original_image: np.ndarray,
    v2_annotated: np.ndarray,
    v2_detections: list[dict],
    v2_texts: list[str],
    v2_det_time: float,
    v2_ocr_times: list[float],
    v1_det_time: float | None,
    v1_ocr_times: list[float] | None,
    trocr_is_ov: bool,
    output_path: Path,
) -> None:
    """Create comparison figure showing V2 OpenVINO results and optional V1 comparison.

    Args:
        original_image: Original input image (BGR format).
        v2_annotated: V2 annotated image (BGR format).
        v2_detections: V2 detection results.
        v2_texts: V2 extracted text strings.
        v2_det_time: V2 detection inference time.
        v2_ocr_times: V2 OCR inference times.
        v1_det_time: V1 detection inference time (None if not run).
        v1_ocr_times: V1 OCR inference times (None if not run).
        trocr_is_ov: Whether TrOCR was running with OpenVINO.
        output_path: Path to save the comparison figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: Original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: OpenVINO detection result
    axes[1].imshow(cv2.cvtColor(v2_annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"OpenVINO Detection\n(Inference: {v2_det_time * 1000:.1f}ms)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].axis("off")

    # Panel 3: Results and comparison
    axes[2].set_facecolor("#f0f0f0")
    v2_total_ocr = sum(v2_ocr_times)
    v2_total = v2_det_time + v2_total_ocr

    text_content = "OpenVINO Optimized Results\n"
    text_content += "=" * 42 + "\n\n"

    if v2_texts:
        for i, (text, det, ocr_time) in enumerate(
            zip(v2_texts, v2_detections, v2_ocr_times)
        ):
            text_content += f"Plate {i + 1}: {text}\n"
            text_content += f"  Conf: {det['confidence']:.4f}  "
            text_content += f"OCR: {ocr_time * 1000:.1f}ms\n"
    else:
        text_content += "No license plates detected.\n"

    text_content += "\n" + "-" * 42 + "\n"
    text_content += "V2 Performance (OpenVINO)\n"
    text_content += "-" * 42 + "\n"
    text_content += f"YOLO Detection:   {v2_det_time * 1000:.1f}ms\n"
    trocr_label = "OV" if trocr_is_ov else "HF"
    text_content += f"TrOCR ({trocr_label}):     {v2_total_ocr * 1000:.1f}ms\n"
    text_content += f"Total Pipeline:    {v2_total * 1000:.1f}ms\n"

    if v1_det_time is not None and v1_ocr_times is not None:
        v1_total_ocr = sum(v1_ocr_times)
        v1_total = v1_det_time + v1_total_ocr
        text_content += "\n" + "-" * 42 + "\n"
        text_content += "V1 Performance (HuggingFace)\n"
        text_content += "-" * 42 + "\n"
        text_content += f"YOLO Detection:   {v1_det_time * 1000:.1f}ms\n"
        text_content += f"TrOCR (HF):       {v1_total_ocr * 1000:.1f}ms\n"
        text_content += f"Total Pipeline:    {v1_total * 1000:.1f}ms\n"

        if v1_total > 0:
            speedup = v1_total / v2_total if v2_total > 0 else 0
            text_content += f"\nSpeedup: {speedup:.2f}x\n"

    axes[2].text(
        0.05,
        0.95,
        text_content,
        transform=axes[2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightcyan", "alpha": 0.8},
    )
    axes[2].set_title(
        "OCR Results & Performance Comparison", fontsize=14, fontweight="bold"
    )
    axes[2].axis("off")

    plt.suptitle(
        "V2: License Plate Detection & Recognition (OpenVINO Optimized)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"Comparison figure saved to: {output_path}")


def process_image(
    image_path: str,
    yolo_ov_model: str = DEFAULT_YOLO_OV_MODEL,
    trocr_ov_model: str = DEFAULT_TROCR_OV_MODEL,
    confidence: float = 0.25,
    output_dir: str = str(OUTPUT_DIR),
    compare_v1: bool = False,
) -> dict:
    """Process an image using OpenVINO optimized models.

    Args:
        image_path: Path to the input car image.
        yolo_ov_model: Path to the OpenVINO YOLO model directory.
        trocr_ov_model: Path to the OpenVINO TrOCR model directory.
        confidence: Detection confidence threshold.
        output_dir: Output directory for results.
        compare_v1: Whether to also run V1 pipeline for comparison.

    Returns:
        Dictionary with detection results and timing information.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)

    print(f"\nProcessing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Load OpenVINO models
    yolo_model = load_yolo_openvino(yolo_ov_model)
    trocr_processor, trocr_model, trocr_is_ov = load_trocr_openvino(trocr_ov_model)

    # V2: Detect license plates with OpenVINO
    print("\nRunning OpenVINO license plate detection...")
    v2_detections, v2_det_time = detect_license_plates_ov(yolo_model, image, confidence)
    print(
        f"Found {len(v2_detections)} plate(s) in {v2_det_time * 1000:.1f}ms (OpenVINO)"
    )

    # V2: Extract text
    v2_texts = []
    v2_ocr_times = []
    for i, detection in enumerate(v2_detections):
        x1, y1, x2, y2 = detection["bbox"]
        plate_crop = image[y1:y2, x1:x2]
        if plate_crop.size == 0:
            v2_texts.append("N/A")
            v2_ocr_times.append(0.0)
            continue

        print(f"\nExtracting text from plate {i + 1} (OpenVINO)...")
        text, ocr_time = extract_text_ov(
            trocr_processor, trocr_model, plate_crop, trocr_is_ov
        )
        v2_texts.append(text)
        v2_ocr_times.append(ocr_time)
        print(f"  Text: '{text}' (OCR time: {ocr_time * 1000:.1f}ms)")

    # Draw detections
    v2_annotated = draw_detections(image, v2_detections, v2_texts)
    annotated_path = output_path / "v2_annotated.jpg"
    cv2.imwrite(str(annotated_path), v2_annotated)

    # Optional: Run V1 pipeline for comparison
    v1_det_time = None
    v1_ocr_times = None
    if compare_v1:
        print("\n--- Running V1 pipeline for comparison ---")
        try:
            _, _, v1_det_time, v1_ocr_times = run_v1_pipeline(image, confidence)
        except Exception as e:
            print(f"V1 comparison failed: {e}")

    # Create comparison figure
    comparison_path = output_path / "v2_comparison.png"
    create_comparison_figure(
        image,
        v2_annotated,
        v2_detections,
        v2_texts,
        v2_det_time,
        v2_ocr_times,
        v1_det_time,
        v1_ocr_times,
        trocr_is_ov,
        comparison_path,
    )

    # Print summary
    v2_total_ocr = sum(v2_ocr_times)
    v2_total = v2_det_time + v2_total_ocr
    print("\n" + "=" * 55)
    print("V2 INFERENCE SUMMARY (OpenVINO Optimized)")
    print("=" * 55)
    print(f"YOLO detection (OpenVINO):   {v2_det_time * 1000:.1f}ms")
    trocr_label = "OpenVINO" if trocr_is_ov else "HuggingFace fallback"
    print(f"TrOCR ({trocr_label}): {v2_total_ocr * 1000:.1f}ms")
    print(f"Total pipeline time:          {v2_total * 1000:.1f}ms")
    print(f"License plates detected:      {len(v2_detections)}")
    for i, (text, conf) in enumerate(
        zip(v2_texts, [d["confidence"] for d in v2_detections])
    ):
        print(f"  Plate {i + 1}: '{text}' (confidence: {conf:.4f})")

    if v1_det_time is not None and v1_ocr_times is not None:
        v1_total = v1_det_time + sum(v1_ocr_times)
        print(f"\nV1 total pipeline time:       {v1_total * 1000:.1f}ms")
        if v2_total > 0:
            print(f"Speedup (V2 vs V1):           {v1_total / v2_total:.2f}x")

    print("=" * 55)

    return {
        "detections": v2_detections,
        "plate_texts": v2_texts,
        "detection_time": v2_det_time,
        "ocr_times": v2_ocr_times,
        "total_time": v2_total,
        "trocr_is_openvino": trocr_is_ov,
    }


def main() -> None:
    """Main entry point for the V2 OpenVINO optimized application."""
    parser = argparse.ArgumentParser(
        description=(
            "V2: License Plate Detection & Text Extraction "
            "using OpenVINO Optimized Models"
        )
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the input car image",
    )
    parser.add_argument(
        "--yolo-ov-model",
        type=str,
        default=DEFAULT_YOLO_OV_MODEL,
        help=f"Path to OpenVINO YOLO model dir (default: {DEFAULT_YOLO_OV_MODEL})",
    )
    parser.add_argument(
        "--trocr-ov-model",
        type=str,
        default=DEFAULT_TROCR_OV_MODEL,
        help=f"Path to OpenVINO TrOCR model dir (default: {DEFAULT_TROCR_OV_MODEL})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--compare-v1",
        action="store_true",
        help="Also run V1 (HuggingFace) pipeline for performance comparison",
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    process_image(
        image_path=args.image,
        yolo_ov_model=args.yolo_ov_model,
        trocr_ov_model=args.trocr_ov_model,
        confidence=args.confidence,
        output_dir=args.output_dir,
        compare_v1=args.compare_v1,
    )


if __name__ == "__main__":
    main()
