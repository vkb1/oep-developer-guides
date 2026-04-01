"""Visualization utilities for side-by-side comparison of results."""

import logging
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def create_stage_comparison(
    input_image: np.ndarray,
    detection_image: np.ndarray,
    ocr_text: str,
    detection_time_ms: float,
    ocr_time_ms: float,
    output_path: str,
    stage_label: str = "Stage 1",
) -> str:
    """Create a side-by-side comparison image showing input, detection, and OCR results.

    Args:
        input_image: Original input image (BGR).
        detection_image: Image with detection bounding boxes (BGR).
        ocr_text: Recognized license plate text.
        detection_time_ms: Detection inference time in milliseconds.
        ocr_time_ms: OCR inference time in milliseconds.
        output_path: Path to save the comparison image.
        stage_label: Label for the pipeline stage.

    Returns:
        Path to the saved comparison image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    detection_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{stage_label} — License Plate Recognition", fontsize=16, fontweight="bold")

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(detection_rgb)
    axes[1].set_title(f"Detection ({detection_time_ms:.1f} ms)")
    axes[1].axis("off")

    axes[2].text(
        0.5,
        0.5,
        ocr_text if ocr_text else "(no text detected)",
        transform=axes[2].transAxes,
        fontsize=20,
        verticalalignment="center",
        horizontalalignment="center",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "edgecolor": "gray"},
    )
    axes[2].set_title(f"OCR Result ({ocr_time_ms:.1f} ms)")
    axes[2].axis("off")

    total_ms = detection_time_ms + ocr_time_ms
    fig.text(
        0.5,
        0.02,
        f"Total inference: {total_ms:.1f} ms  |  Detection: {detection_time_ms:.1f} ms  |"
        f"  OCR: {ocr_time_ms:.1f} ms",
        ha="center",
        fontsize=11,
        style="italic",
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Comparison saved to %s", output_path)
    return output_path


def create_video_comparison(
    input_frame: np.ndarray,
    annotated_frame: np.ndarray,
    fps: float,
    output_path: str,
    stage_label: str = "Stage 3",
) -> str:
    """Create a side-by-side comparison of input and annotated video frames.

    Args:
        input_frame: Original video frame (BGR).
        annotated_frame: Annotated video frame with detections (BGR).
        fps: Frames per second of the pipeline.
        output_path: Path to save the comparison image.
        stage_label: Label for the pipeline stage.

    Returns:
        Path to the saved comparison image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    input_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"{stage_label} — DL Streamer Pipeline ({fps:.1f} FPS)",
        fontsize=16,
        fontweight="bold",
    )

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input Stream")
    axes[0].axis("off")

    axes[1].imshow(annotated_rgb)
    axes[1].set_title("Annotated Output")
    axes[1].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Video comparison saved to %s", output_path)
    return output_path
