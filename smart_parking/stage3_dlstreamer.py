"""Stage 3: DL Streamer pipeline for license plate recognition on video.

Constructs a GStreamer / DL Streamer pipeline adapted from the
open-edge-platform/dlstreamer license_plate_recognition sample to process
a video stream, detect license plates, recognize text, and display
input/output streams side-by-side with FPS metrics.
"""

import logging
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from smart_parking.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DLSTREAMER_OUTPUT_DIR,
    OUTPUT_DIR,
    OV_YOLO_MODEL_PATH,
    ensure_directories,
)
from smart_parking.stage1_inference import _crop_plates, _recognize_plate_text
from smart_parking.stage2_openvino import _recognize_with_openvino
from smart_parking.visualization import (
    create_video_comparison_figure,
    draw_detections,
)

logger = logging.getLogger(__name__)


def _check_gstreamer_available() -> bool:
    """Check if GStreamer and DL Streamer are available on the system.

    Returns:
        True if GStreamer is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["gst-launch-1.0", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_dlstreamer_pipeline(
    input_video: str,
    detection_model: str,
    output_video: str | None = None,
    device: str = DEFAULT_DEVICE,
) -> str:
    """Build a DL Streamer GStreamer pipeline string for license plate recognition.

    Adapted from the open-edge-platform/dlstreamer license_plate_recognition
    sample pipeline.

    Args:
        input_video: Path to input video file.
        detection_model: Path to the OpenVINO detection model XML.
        output_video: Optional path for output video file.
        device: Inference device (CPU, GPU, etc.).

    Returns:
        GStreamer pipeline string.
    """
    pipeline_parts = [
        f"filesrc location={input_video}",
        "! decodebin",
        "! videoconvert",
        "! video/x-raw,format=BGRx",
        f"! gvadetect model={detection_model} device={device}"
        f" threshold={DEFAULT_CONFIDENCE_THRESHOLD}",
        "! queue",
        "! gvawatermark",
    ]

    if output_video:
        pipeline_parts.extend([
            "! videoconvert",
            "! x264enc",
            f"! filesink location={output_video}",
        ])
    else:
        pipeline_parts.extend([
            "! videoconvert",
            "! fpsdisplaysink video-sink=fakesink sync=false",
        ])

    return " ".join(pipeline_parts)


def _run_dlstreamer_pipeline(pipeline_str: str) -> dict:
    """Execute a DL Streamer pipeline using gst-launch-1.0.

    Args:
        pipeline_str: The GStreamer pipeline string to execute.

    Returns:
        Dict with 'returncode', 'stdout', 'stderr', and 'duration_s'.
    """
    logger.info("Running DL Streamer pipeline:\n  %s", pipeline_str)
    start = time.perf_counter()

    result = subprocess.run(
        ["gst-launch-1.0", "-e"] + pipeline_str.split(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    duration = time.perf_counter() - start

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_s": duration,
    }


def _process_video_opencv(
    input_video: str,
    model: YOLO,
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    output_dir: Path = DLSTREAMER_OUTPUT_DIR,
    use_openvino_ocr: bool = True,
    max_frames: int = 0,
) -> dict:
    """Process video using OpenCV with YOLO detection and OCR as a fallback
    when GStreamer/DL Streamer is not available.

    Reads each frame, runs detection and OCR, and produces side-by-side
    output with FPS metrics.

    Args:
        input_video: Path to input video file.
        model: Loaded YOLO model (native or OpenVINO).
        confidence: Detection confidence threshold.
        output_dir: Directory for output files.
        use_openvino_ocr: Whether to use OpenVINO-backed OCR.
        max_frames: Maximum frames to process (0 = all).

    Returns:
        Dict with processing results and metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Video: %s, %dx%d, %.1f FPS, %d frames",
        input_video,
        width,
        height,
        video_fps,
        total_frames,
    )

    output_video_path = output_dir / "annotated_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, video_fps, (width, height))

    frame_count = 0
    total_det_time = 0.0
    total_ocr_time = 0.0
    all_texts = []
    fps_values = []
    sample_input = None
    sample_output = None

    process_limit = max_frames if max_frames > 0 else total_frames

    while cap.isOpened() and frame_count < process_limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()

        # Detection
        det_start = time.perf_counter()
        results = model.predict(frame, conf=confidence, verbose=False)
        det_time = (time.perf_counter() - det_start) * 1000
        total_det_time += det_time

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

        annotated = draw_detections(frame, detections)

        # OCR on detected plates (every 10th frame to keep performance)
        if detections and frame_count % 10 == 0:
            crops = _crop_plates(frame, detections)
            if use_openvino_ocr:
                texts, ocr_time = _recognize_with_openvino(crops)
            else:
                texts, ocr_time = _recognize_plate_text(crops)
            total_ocr_time += ocr_time
            all_texts.extend(texts)

        frame_time = (time.perf_counter() - frame_start) * 1000
        current_fps = 1000.0 / frame_time if frame_time > 0 else 0
        fps_values.append(current_fps)

        # Add FPS overlay
        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        writer.write(annotated)

        # Capture sample frames for comparison
        if frame_count == min(30, process_limit - 1):
            sample_input = frame.copy()
            sample_output = annotated.copy()

        frame_count += 1

        if frame_count % 50 == 0:
            avg_fps = sum(fps_values[-50:]) / min(50, len(fps_values))
            logger.info(
                "Processed %d/%d frames (avg FPS: %.1f)",
                frame_count,
                process_limit,
                avg_fps,
            )

    cap.release()
    writer.release()

    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

    # Create comparison figure
    if sample_input is not None and sample_output is not None:
        create_video_comparison_figure(
            input_frame=sample_input,
            output_frame=sample_output,
            fps=avg_fps,
            frame_number=min(30, frame_count - 1),
            output_path=output_dir / "video_comparison.png",
        )

    logger.info(
        "Video processing complete: %d frames, avg FPS: %.1f",
        frame_count,
        avg_fps,
    )

    return {
        "frames_processed": frame_count,
        "avg_fps": avg_fps,
        "avg_detection_ms": total_det_time / max(frame_count, 1),
        "total_ocr_ms": total_ocr_time,
        "unique_plates": list(set(all_texts)),
        "output_video": str(output_video_path),
        "comparison_image": str(output_dir / "video_comparison.png"),
    }


def run_stage3(
    input_video: str,
    ov_yolo_model: str | None = None,
    device: str = DEFAULT_DEVICE,
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_frames: int = 0,
    output_dir: str | None = None,
) -> dict:
    """Execute Stage 3: DL Streamer pipeline for video processing.

    Attempts to use GStreamer/DL Streamer if available, otherwise falls
    back to an OpenCV-based pipeline that mimics the DL Streamer behavior.

    Args:
        input_video: Path to input video file.
        ov_yolo_model: Path to OpenVINO YOLO model XML (from Stage 2).
        device: Inference device (CPU, GPU, etc.).
        confidence: Detection confidence threshold.
        max_frames: Maximum frames to process (0 = all).
        output_dir: Optional output directory.

    Returns:
        Dict with pipeline execution results and metrics.
    """
    ensure_directories()
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(input_video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    model_path = ov_yolo_model or str(OV_YOLO_MODEL_PATH)

    use_dlstreamer = _check_gstreamer_available()

    if use_dlstreamer:
        logger.info("DL Streamer available - using GStreamer pipeline")
        output_video = str(out_dir / "annotated_output.mp4")
        pipeline = _build_dlstreamer_pipeline(
            input_video=str(video_path),
            detection_model=model_path,
            output_video=output_video,
            device=device,
        )

        result = _run_dlstreamer_pipeline(pipeline)

        if result["returncode"] != 0:
            logger.warning(
                "DL Streamer pipeline failed (rc=%d), falling back to OpenCV",
                result["returncode"],
            )
            use_dlstreamer = False
        else:
            logger.info(
                "DL Streamer pipeline completed in %.1f s",
                result["duration_s"],
            )
            return {
                "backend": "dlstreamer",
                "pipeline": pipeline,
                "duration_s": result["duration_s"],
                "output_video": output_video,
                "stdout": result["stdout"],
            }

    if not use_dlstreamer:
        logger.info(
            "DL Streamer not available - using OpenCV pipeline with "
            "YOLO + OCR (mimicking DL Streamer behavior)"
        )

        model = YOLO(model_path)

        return {
            "backend": "opencv",
            **_process_video_opencv(
                input_video=str(video_path),
                model=model,
                confidence=confidence,
                output_dir=out_dir,
                use_openvino_ocr=True,
                max_frames=max_frames,
            ),
        }
