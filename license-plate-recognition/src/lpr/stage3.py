"""Stage 3: DL Streamer pipeline for license plate recognition on video streams.

Constructs and runs a GStreamer/DL Streamer pipeline adapted from the
open-edge-platform/dlstreamer license_plate_recognition sample. Requires
DL Streamer and GStreamer to be installed on the system.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time

import cv2
import numpy as np

from . import config
from .stage2 import convert_models
from .visualization import create_video_comparison

logger = logging.getLogger(__name__)


def _check_dlstreamer_available() -> bool:
    """Check if DL Streamer / GStreamer is available on the system."""
    return shutil.which("gst-launch-1.0") is not None


def _build_source_element(input_path: str) -> str:
    """Build the GStreamer source element based on input type.

    Args:
        input_path: Path or URL to input video.

    Returns:
        GStreamer source element string.
    """
    if input_path.startswith("/dev/video"):
        return f"v4l2src device={input_path}"
    elif "://" in input_path:
        return f"urisourcebin buffer-size=4096 uri={input_path}"
    else:
        return f"filesrc location={input_path}"


def _build_decode_element(device: str) -> str:
    """Build the GStreamer decode element based on device.

    Args:
        device: Inference device (CPU or GPU).

    Returns:
        GStreamer decode element string.
    """
    if device == "GPU":
        return r"decodebin3 ! vapostproc ! video/x-raw\(memory:VAMemory\)"
    return "decodebin3"


def _build_preprocess_arg(device: str) -> str:
    """Build the pre-process backend argument.

    Args:
        device: Inference device.

    Returns:
        Pre-process backend string.
    """
    if device == "GPU":
        return "pre-process-backend=va-surface-sharing"
    return "pre-process-backend=opencv"


def build_pipeline(
    input_path: str,
    detection_model_path: str,
    ocr_model_path: str,
    device: str = config.DEFAULT_DEVICE,
    output_mode: str = "fps",
    output_file: str = "output.json",
) -> str:
    """Build the DL Streamer GStreamer pipeline command string.

    Adapts the license_plate_recognition.sh pipeline from the DL Streamer
    samples repository for programmatic use.

    Args:
        input_path: Path or URL to input video/stream.
        detection_model_path: Path to detection model IR .xml file.
        ocr_model_path: Path to OCR model IR .xml file.
        device: Inference device (CPU, GPU, AUTO).
        output_mode: Output mode (display, fps, json, display-and-json, file).
        output_file: Output file path for json/file modes.

    Returns:
        Complete gst-launch-1.0 command string.
    """
    source = _build_source_element(input_path)
    decode = _build_decode_element(device)
    preproc = _build_preprocess_arg(device)

    if output_mode == "display":
        sink = "gvawatermark ! videoconvert ! gvafpscounter ! autovideosink"
    elif output_mode == "fps":
        sink = "gvafpscounter ! fakesink async=false"
    elif output_mode == "json":
        sink = (
            f"gvametaconvert ! gvametapublish file-format=json-lines "
            f"file-path={output_file} ! fakesink async=false"
        )
    elif output_mode == "display-and-json":
        sink = (
            f"gvawatermark ! gvametaconvert ! gvametapublish file-format=json-lines "
            f"file-path={output_file} ! videoconvert ! gvafpscounter ! autovideosink sync=false"
        )
    elif output_mode == "file":
        sink = (
            "gvawatermark ! gvafpscounter ! videoconvert ! x264enc ! "
            f"h264parse ! mp4mux ! filesink location={output_file}"
        )
    else:
        raise ValueError(
            f"Unsupported output mode: {output_mode}. "
            "Use: display, fps, json, display-and-json, file"
        )

    pipeline = (
        f"gst-launch-1.0 {source} ! {decode} ! queue ! "
        f"gvadetect model={detection_model_path} device={device} {preproc} ! queue ! "
        f"videoconvert ! "
        f"gvaclassify model={ocr_model_path} device={device} {preproc} ! queue ! "
        f"{sink}"
    )

    return pipeline


def _parse_fps_from_output(output: str) -> float:
    """Extract FPS value from DL Streamer pipeline output.

    Args:
        output: Combined stdout/stderr output from the pipeline.

    Returns:
        Extracted FPS value, or 0.0 if not found.
    """
    fps_pattern = re.compile(r"FpsCounter.*?average\s*=\s*([\d.]+)", re.IGNORECASE)
    match = fps_pattern.search(output)
    if match:
        return float(match.group(1))

    # Fallback: look for any fps-like pattern
    fps_pattern2 = re.compile(r"([\d.]+)\s*fps", re.IGNORECASE)
    match = fps_pattern2.search(output)
    if match:
        return float(match.group(1))

    return 0.0


def run(
    input_path: str | None = None,
    models_dir: str = config.DEFAULT_MODELS_DIR,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
    device: str = config.DEFAULT_DEVICE,
    output_mode: str = "json",
) -> dict:
    """Run Stage 3: DL Streamer pipeline for video stream processing.

    Args:
        input_path: Path or URL to input video. Defaults to the sample parking video.
        models_dir: Directory containing downloaded/converted models.
        output_dir: Directory for saving output.
        device: Inference device (CPU, GPU, AUTO).
        output_mode: DL Streamer output mode.

    Returns:
        Dictionary with pipeline results and FPS information.
    """
    logger.info("=== Stage 3: DL Streamer Pipeline ===")

    if not _check_dlstreamer_available():
        logger.error(
            "DL Streamer (gst-launch-1.0) is not available. "
            "Please install DL Streamer: https://dlstreamer.github.io/get_started/install.html"
        )
        return {
            "stage": 3,
            "error": "DL Streamer not installed",
            "pipeline": None,
            "fps": 0.0,
        }

    # Ensure models are converted to OpenVINO IR
    det_ir_path, ocr_ir_path = convert_models(models_dir)

    # Use default video if no input specified
    if input_path is None:
        input_path = config.DEFAULT_VIDEO_URL

    os.makedirs(output_dir, exist_ok=True)

    # Determine output file path
    json_output = os.path.join(output_dir, "stage3_detections.json")
    video_output = os.path.join(output_dir, "stage3_annotated.mp4")

    if output_mode == "file":
        out_file = video_output
    elif output_mode in ("json", "display-and-json"):
        out_file = json_output
        # Remove old output file
        if os.path.exists(json_output):
            os.remove(json_output)
    else:
        out_file = json_output

    # Build and run pipeline
    pipeline_cmd = build_pipeline(
        input_path=input_path,
        detection_model_path=det_ir_path,
        ocr_model_path=ocr_ir_path,
        device=device,
        output_mode=output_mode,
        output_file=out_file,
    )

    logger.info("Pipeline command:\n%s", pipeline_cmd)

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            pipeline_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=config.PIPELINE_TIMEOUT_SECONDS,
        )
        elapsed_s = time.perf_counter() - start_time

        combined_output = result.stdout + "\n" + result.stderr
        logger.info("Pipeline output:\n%s", combined_output)

        fps = _parse_fps_from_output(combined_output)
        if fps == 0.0 and elapsed_s > 0:
            logger.info("FPS not reported by pipeline; elapsed time: %.1f s", elapsed_s)

    except subprocess.TimeoutExpired:
        logger.warning(
            "Pipeline timed out after %d seconds", config.PIPELINE_TIMEOUT_SECONDS
        )
        fps = 0.0
        combined_output = ""

    # --- Generate comparison visualization ---
    comparison_path = os.path.join(output_dir, "stage3_comparison.png")

    # Try to extract frames for comparison
    _generate_comparison(input_path, json_output, video_output, fps, comparison_path, output_mode)

    return {
        "stage": 3,
        "pipeline": pipeline_cmd,
        "fps": fps,
        "output_mode": output_mode,
        "output_path": comparison_path,
    }


def _generate_comparison(
    input_path: str,
    json_output: str,
    video_output: str,
    fps: float,
    comparison_path: str,
    output_mode: str,
) -> None:
    """Generate a visual comparison from pipeline results.

    Args:
        input_path: Original input video path/URL.
        json_output: Path to JSON detection output.
        video_output: Path to annotated video output.
        fps: Pipeline FPS value.
        comparison_path: Where to save the comparison image.
        output_mode: The output mode used for the pipeline.
    """
    # Get a frame from the input video
    input_frame = _get_video_frame(input_path)
    if input_frame is None:
        logger.warning("Could not extract frame from input video for comparison")
        return

    # Try to get annotated frame
    annotated_frame = None
    if output_mode == "file" and os.path.exists(video_output):
        annotated_frame = _get_video_frame(video_output)

    # If no annotated video, create annotation from JSON metadata
    if annotated_frame is None and os.path.exists(json_output):
        annotated_frame = _annotate_from_json(input_frame, json_output)

    if annotated_frame is None:
        annotated_frame = input_frame.copy()
        cv2.putText(
            annotated_frame,
            f"Pipeline FPS: {fps:.1f}" if fps > 0 else "Pipeline completed",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    create_video_comparison(
        input_frame=input_frame,
        annotated_frame=annotated_frame,
        fps=fps,
        output_path=comparison_path,
    )


def _get_video_frame(video_path: str, frame_idx: int = 30) -> np.ndarray | None:
    """Extract a single frame from a video file or URL.

    Args:
        video_path: Path or URL to the video.
        frame_idx: Index of the frame to extract.

    Returns:
        Frame as BGR numpy array, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def _annotate_from_json(frame: np.ndarray, json_path: str) -> np.ndarray:
    """Draw detection annotations on a frame using JSON metadata.

    Args:
        frame: Input frame to annotate (BGR).
        json_path: Path to the DL Streamer JSON-lines output file.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    try:
        with open(json_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                objects = data.get("objects", [])
                for obj in objects:
                    detection = obj.get("detection", {})
                    bbox = detection.get("bounding_box", {})

                    x_min = int(bbox.get("x_min", 0) * w)
                    y_min = int(bbox.get("y_min", 0) * h)
                    x_max = int(bbox.get("x_max", 0) * w)
                    y_max = int(bbox.get("y_max", 0) * h)

                    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    label = detection.get("label", "plate")
                    confidence = detection.get("confidence", 0.0)
                    text = f"{label} {confidence:.2f}"
                    cv2.putText(
                        annotated,
                        text,
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                # Only process first frame's worth of detections
                break
    except (OSError, ValueError) as e:
        logger.warning("Could not parse JSON output: %s", e)

    return annotated
