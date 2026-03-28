"""Visualization utilities for side-by-side comparison of results."""

import logging
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smart_parking.config import FIGURE_DPI, FIGURE_SIZE, OUTPUT_DIR

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def draw_detections(
    image: np.ndarray,
    boxes: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: Input image (BGR format).
        boxes: List of detection dicts with keys 'bbox', 'confidence',
            and optionally 'label'.
        color: BGR color for the bounding box.
        thickness: Line thickness for drawing.

    Returns:
        Image with drawn detections.
    """
    annotated = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box["bbox"]]
        conf = box.get("confidence", 0.0)
        label = box.get("label", "")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        text = f"{label} {conf:.2f}" if label else f"{conf:.2f}"
        font_scale = 0.6
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
    return annotated


def create_comparison_figure(
    original: np.ndarray,
    detection: np.ndarray,
    ocr_texts: list[str],
    inference_times: dict,
    title: str = "License Plate Detection & Recognition",
    output_path: Path | None = None,
) -> Path:
    """Create a side-by-side comparison figure.

    Args:
        original: Original input image (BGR).
        detection: Image with detection boxes drawn (BGR).
        ocr_texts: List of recognized text strings from detected plates.
        inference_times: Dict with timing keys like 'detection_ms',
            'ocr_ms', 'total_ms'.
        title: Figure title.
        output_path: Path to save the figure. Auto-generated if None.

    Returns:
        Path to the saved comparison figure.
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    detection_rgb = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(original_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    det_time = inference_times.get("detection_ms", 0)
    axes[1].imshow(detection_rgb)
    axes[1].set_title(f"License Plate Detection\n({det_time:.1f} ms)")
    axes[1].axis("off")

    ocr_time = inference_times.get("ocr_ms", 0)
    total_time = inference_times.get("total_ms", 0)

    text_content = "Recognized Plates:\n" + "-" * 30 + "\n"
    if ocr_texts:
        for i, text in enumerate(ocr_texts, 1):
            text_content += f"\n  Plate {i}: {text}\n"
    else:
        text_content += "\n  No plates detected\n"

    text_content += "\n" + "-" * 30
    text_content += f"\nOCR Time: {ocr_time:.1f} ms"
    text_content += f"\nTotal Time: {total_time:.1f} ms"

    axes[2].text(
        0.5,
        0.5,
        text_content,
        transform=axes[2].transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.8},
    )
    axes[2].set_title(f"OCR Results ({ocr_time:.1f} ms)")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)

    logger.info("Comparison figure saved to %s", output_path)
    return output_path


def create_video_comparison_figure(
    input_frame: np.ndarray,
    output_frame: np.ndarray,
    fps: float,
    frame_number: int,
    output_path: Path | None = None,
) -> Path:
    """Create a side-by-side comparison of input and annotated video frames.

    Args:
        input_frame: Original video frame (BGR).
        output_frame: Annotated video frame (BGR).
        fps: Current processing FPS.
        frame_number: Current frame number.
        output_path: Path to save the figure. Auto-generated if None.

    Returns:
        Path to the saved comparison figure.
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "video_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI)
    fig.suptitle(
        f"DL Streamer Pipeline - Frame {frame_number} | FPS: {fps:.1f}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input Stream")
    axes[0].axis("off")

    axes[1].imshow(output_rgb)
    axes[1].set_title("Annotated Output Stream")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)

    logger.info("Video comparison saved to %s", output_path)
    return output_path
