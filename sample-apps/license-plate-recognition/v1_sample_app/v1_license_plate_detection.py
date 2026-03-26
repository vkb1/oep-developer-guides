"""V1 Sample App: License Plate Detection and Text Extraction using HuggingFace Models.

This application uses:
- YOLOv11 (morsetechlab/yolov11-license-plate-detection) for license plate detection
- TrOCR (microsoft/trocr-base-printed) for text extraction from detected plates

It displays a side-by-side comparison of:
1. Original input image
2. License plate detection with bounding boxes
3. Extracted text with inference timing numbers
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

# Use non-interactive backend when no display is available
matplotlib.use("Agg")

# Default paths
DEFAULT_YOLO_MODEL = "morsetechlab/yolov11-license-plate-detection"
DEFAULT_TROCR_MODEL = "microsoft/trocr-base-printed"
OUTPUT_DIR = Path("output")


def load_yolo_model(model_path: str) -> YOLO:
    """Load the YOLOv11 license plate detection model.

    Args:
        model_path: HuggingFace model ID or local path to the YOLO model.

    Returns:
        Loaded YOLO model instance.
    """
    print(f"Loading YOLO model from: {model_path}")
    start_time = time.time()
    model = YOLO(model_path)
    load_time = time.time() - start_time
    print(f"YOLO model loaded in {load_time:.2f}s")
    return model


def load_trocr_model(
    model_path: str,
) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the TrOCR model and processor for text recognition.

    Args:
        model_path: HuggingFace model ID or local path to the TrOCR model.

    Returns:
        Tuple of (processor, model) for TrOCR inference.
    """
    print(f"Loading TrOCR model from: {model_path}")
    start_time = time.time()
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    load_time = time.time() - start_time
    print(f"TrOCR model loaded in {load_time:.2f}s")
    return processor, model


def detect_license_plates(
    model: YOLO, image: np.ndarray, confidence: float = 0.25
) -> tuple[list[dict], float]:
    """Detect license plates in the given image using YOLO.

    Args:
        model: Loaded YOLO model.
        image: Input image as numpy array (BGR format).
        confidence: Minimum confidence threshold for detections.

    Returns:
        Tuple of (list of detection dicts, inference time in seconds).
        Each detection dict contains 'bbox', 'confidence', and 'class' keys.
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


def extract_text_from_plate(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    plate_image: np.ndarray,
) -> tuple[str, float]:
    """Extract text from a license plate image using TrOCR.

    Args:
        processor: TrOCR processor for image preprocessing.
        model: TrOCR model for text generation.
        plate_image: Cropped license plate image as numpy array (BGR format).

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

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
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


def create_comparison_figure(
    original_image: np.ndarray,
    annotated_image: np.ndarray,
    detections: list[dict],
    plate_texts: list[str],
    detection_time: float,
    ocr_times: list[float],
    output_path: Path,
) -> None:
    """Create and save a side-by-side comparison figure.

    Args:
        original_image: Original input image (BGR format).
        annotated_image: Image with detection annotations (BGR format).
        detections: List of detection dicts.
        plate_texts: List of extracted text strings.
        detection_time: YOLO detection inference time in seconds.
        ocr_times: List of OCR inference times for each plate.
        output_path: Path to save the comparison figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Detection result
    axes[1].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"License Plate Detection\n(Inference: {detection_time * 1000:.1f}ms)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].axis("off")

    # Panel 3: Extracted text and inference details
    axes[2].set_facecolor("#f0f0f0")
    text_content = "Extracted License Plate Text\n"
    text_content += "=" * 40 + "\n\n"

    total_ocr_time = sum(ocr_times)

    if plate_texts:
        for i, (text, detection, ocr_time) in enumerate(
            zip(plate_texts, detections, ocr_times)
        ):
            conf = detection["confidence"]
            text_content += f"Plate {i + 1}:\n"
            text_content += f"  Text: {text}\n"
            text_content += f"  Confidence: {conf:.4f}\n"
            text_content += f"  OCR Time: {ocr_time * 1000:.1f}ms\n\n"
    else:
        text_content += "No license plates detected.\n\n"

    text_content += "-" * 40 + "\n"
    text_content += "Performance Summary\n"
    text_content += "-" * 40 + "\n"
    text_content += f"Detection Time: {detection_time * 1000:.1f}ms\n"
    text_content += f"Total OCR Time: {total_ocr_time * 1000:.1f}ms\n"
    text_content += (
        f"Total Pipeline: {(detection_time + total_ocr_time) * 1000:.1f}ms\n"
    )
    text_content += f"Plates Found: {len(detections)}\n"

    axes[2].text(
        0.05,
        0.95,
        text_content,
        transform=axes[2].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )
    axes[2].set_title("OCR Results & Inference Stats", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        "V1: License Plate Detection & Recognition (HuggingFace Models)",
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
    yolo_model_path: str = DEFAULT_YOLO_MODEL,
    trocr_model_path: str = DEFAULT_TROCR_MODEL,
    confidence: float = 0.25,
    output_dir: str = str(OUTPUT_DIR),
) -> dict:
    """Process a single image for license plate detection and text extraction.

    Args:
        image_path: Path to the input car image.
        yolo_model_path: HuggingFace model ID or local path for YOLO model.
        trocr_model_path: HuggingFace model ID or local path for TrOCR model.
        confidence: Minimum detection confidence threshold.
        output_dir: Directory to save output files.

    Returns:
        Dictionary containing detection results and timing information.
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

    # Load models
    yolo_model = load_yolo_model(yolo_model_path)
    trocr_processor, trocr_model = load_trocr_model(trocr_model_path)

    # Detect license plates
    print("\nRunning license plate detection...")
    detections, detection_time = detect_license_plates(
        yolo_model, image, confidence
    )
    print(f"Found {len(detections)} license plate(s) in {detection_time * 1000:.1f}ms")

    # Extract text from each detected plate
    plate_texts = []
    ocr_times = []
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox"]
        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            plate_texts.append("N/A")
            ocr_times.append(0.0)
            continue

        print(f"\nExtracting text from plate {i + 1}...")
        text, ocr_time = extract_text_from_plate(trocr_processor, trocr_model, plate_crop)
        plate_texts.append(text)
        ocr_times.append(ocr_time)
        print(f"  Text: '{text}' (OCR time: {ocr_time * 1000:.1f}ms)")

    # Draw detections on image
    annotated_image = draw_detections(image, detections, plate_texts)

    # Save annotated image
    annotated_path = output_path / "v1_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated_image)
    print(f"\nAnnotated image saved to: {annotated_path}")

    # Create comparison figure
    comparison_path = output_path / "v1_comparison.png"
    create_comparison_figure(
        image,
        annotated_image,
        detections,
        plate_texts,
        detection_time,
        ocr_times,
        comparison_path,
    )

    # Print summary
    total_ocr_time = sum(ocr_times)
    print("\n" + "=" * 50)
    print("V1 INFERENCE SUMMARY")
    print("=" * 50)
    print(f"Detection inference time:  {detection_time * 1000:.1f}ms")
    print(f"Total OCR inference time:  {total_ocr_time * 1000:.1f}ms")
    print(f"Total pipeline time:       {(detection_time + total_ocr_time) * 1000:.1f}ms")
    print(f"License plates detected:   {len(detections)}")
    for i, (text, conf) in enumerate(
        zip(plate_texts, [d["confidence"] for d in detections])
    ):
        print(f"  Plate {i + 1}: '{text}' (confidence: {conf:.4f})")
    print("=" * 50)

    return {
        "detections": detections,
        "plate_texts": plate_texts,
        "detection_time": detection_time,
        "ocr_times": ocr_times,
        "total_time": detection_time + total_ocr_time,
    }


def main() -> None:
    """Main entry point for the V1 license plate detection application."""
    parser = argparse.ArgumentParser(
        description="V1: License Plate Detection & Text Extraction using HuggingFace Models"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the input car image",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=DEFAULT_YOLO_MODEL,
        help=f"YOLO model path or HuggingFace ID (default: {DEFAULT_YOLO_MODEL})",
    )
    parser.add_argument(
        "--trocr-model",
        type=str,
        default=DEFAULT_TROCR_MODEL,
        help=f"TrOCR model path or HuggingFace ID (default: {DEFAULT_TROCR_MODEL})",
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

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    process_image(
        image_path=args.image,
        yolo_model_path=args.yolo_model,
        trocr_model_path=args.trocr_model,
        confidence=args.confidence,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
