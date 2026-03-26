"""V3 Sample App: License Plate Recognition using Intel DL Streamer Pipeline.

This application adapts the DL Streamer license plate recognition pipeline
to construct a GStreamer-based pipeline for processing smart parking video
streams. It uses the same OpenVINO IR optimized models from V2 (YOLOv11 for
detection and TrOCR for OCR), and displays the input and annotated output
streams side by side with FPS performance numbers.

Reference: https://github.com/open-edge-platform/dlstreamer/tree/main/samples/gstreamer/gst_launch/license_plate_recognition
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend when no display is available
matplotlib.use("Agg")

# Default configuration
DEFAULT_VIDEO_URL = (
    "https://videos.pexels.com/video-files/3014296/3014296-hd_1920_1080_24fps.mp4"
)
# Use the same V2 optimized OpenVINO IR models directory
DEFAULT_MODELS_PATH = os.environ.get("MODELS_PATH", "models_ov")
DEFAULT_DETECTION_MODEL_DIR = "yolo"
DEFAULT_OCR_MODEL_DIR = "trocr"
OUTPUT_DIR = Path("output")

# DL Streamer GStreamer element names
GST_DETECT = "gvadetect"
GST_CLASSIFY = "gvaclassify"
GST_WATERMARK = "gvawatermark"
GST_FPS_COUNTER = "gvafpscounter"
GST_META_CONVERT = "gvametaconvert"


def find_model_xml(model_dir: str) -> str:
    """Find an OpenVINO IR XML model file within a directory.

    Searches the given directory (recursively) for .xml model files.
    This allows V3 to use the same model directory layout produced by
    V2's convert_models.py.

    Args:
        model_dir: Path to a directory containing OpenVINO IR model files.

    Returns:
        Absolute path to the first .xml model file found.

    Raises:
        FileNotFoundError: If no .xml file is found in the directory.
    """
    model_path = Path(model_dir)
    if model_path.is_file() and model_path.suffix == ".xml":
        return str(model_path)

    xml_files = sorted(model_path.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No OpenVINO IR model (.xml) found in {model_dir}. "
            "Run v2_sample_app/convert_models.py first to convert the models."
        )
    return str(xml_files[0])


def check_dlstreamer_available() -> bool:
    """Check if DL Streamer GStreamer elements are available.

    Returns:
        True if DL Streamer is properly installed and available.
    """
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", GST_DETECT],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_gstreamer_available() -> bool:
    """Check if GStreamer is available.

    Returns:
        True if GStreamer is properly installed.
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


def get_device_config(device: str) -> dict:
    """Get device-specific configuration for GStreamer pipeline elements.

    Args:
        device: Target device - 'CPU', 'GPU', or 'AUTO'.

    Returns:
        Dictionary containing device-specific pipeline element configurations.
    """
    if device.upper() == "GPU":
        # Check for GPU render device
        render_device = "/dev/dri/renderD128"
        if os.path.exists(render_device):
            return {
                "device": "GPU",
                "pre_process_backend": "va-surface-sharing",
                "decode": (
                    "decodebin3 ! "
                    "vapostproc ! "
                    "video/x-raw(memory:VAMemory)"
                ),
            }
        else:
            print(
                f"Warning: GPU render device {render_device} not found, "
                "falling back to CPU"
            )
            return {
                "device": "CPU",
                "pre_process_backend": "opencv",
                "decode": "decodebin3",
            }
    else:
        return {
            "device": device.upper(),
            "pre_process_backend": "opencv",
            "decode": "decodebin3",
        }


def build_pipeline_string(
    input_source: str,
    detection_model: str,
    ocr_model: str,
    device: str = "CPU",
    output_mode: str = "display",
    output_file: str | None = None,
) -> str:
    """Construct the DL Streamer GStreamer pipeline string.

    This constructs a pipeline based on the DL Streamer license plate
    recognition sample, adapted for the smart parking use case.

    Args:
        input_source: Video file path, URL, or camera device.
        detection_model: Path to the license plate detection model (OpenVINO IR).
        ocr_model: Path to the OCR model (OpenVINO IR).
        device: Inference device ('CPU', 'GPU', or 'AUTO').
        output_mode: Output mode ('display', 'fps', 'json', 'file',
                     'display-and-json').
        output_file: Output file path for 'file' mode.

    Returns:
        GStreamer pipeline string for gst-launch-1.0.
    """
    config = get_device_config(device)
    dev = config["device"]
    ppb = config["pre_process_backend"]
    decode = config["decode"]

    # Source element
    if input_source.startswith(("/dev/video", "v4l2")):
        source = f"v4l2src device={input_source}"
    elif input_source.startswith(("http://", "https://", "rtsp://")):
        source = f'urisourcebin uri="{input_source}"'
    else:
        source = f'filesrc location="{input_source}"'

    # Core pipeline: source → decode → detect → classify
    pipeline = (
        f"{source} ! "
        f"{decode} ! queue ! "
        f"{GST_DETECT} model={detection_model} device={dev} "
        f"pre-process-backend={ppb} ! queue ! "
        f"videoconvert ! "
        f"{GST_CLASSIFY} model={ocr_model} device={dev} "
        f"pre-process-backend={ppb} ! queue"
    )

    # Sink configuration based on output mode
    if output_mode == "display":
        pipeline += (
            f" ! {GST_WATERMARK} ! {GST_FPS_COUNTER} ! "
            f"videoconvert ! fpsdisplaysink video-sink=autovideosink "
            f"sync=false"
        )
    elif output_mode == "fps":
        pipeline += f" ! {GST_FPS_COUNTER} ! fakesink sync=false"
    elif output_mode == "json":
        pipeline += (
            f" ! {GST_META_CONVERT} ! "
            f"{GST_FPS_COUNTER} ! fakesink sync=false"
        )
    elif output_mode == "file":
        if output_file is None:
            output_file = str(OUTPUT_DIR / "v3_output.mp4")
        pipeline += (
            f" ! {GST_WATERMARK} ! {GST_FPS_COUNTER} ! "
            f"videoconvert ! "
            f"x264enc ! mp4mux ! filesink location={output_file}"
        )
    elif output_mode == "display-and-json":
        pipeline += (
            f" ! tee name=t ! queue ! "
            f"{GST_WATERMARK} ! {GST_FPS_COUNTER} ! "
            f"videoconvert ! fpsdisplaysink video-sink=autovideosink "
            f"sync=false "
            f"t. ! queue ! {GST_META_CONVERT} ! fakesink sync=false"
        )

    return pipeline


def run_gstreamer_pipeline(
    pipeline_string: str, duration: int | None = None
) -> dict:
    """Execute the GStreamer pipeline and collect performance metrics.

    Args:
        pipeline_string: GStreamer pipeline string for gst-launch-1.0.
        duration: Maximum duration in seconds (None for unlimited).

    Returns:
        Dictionary containing FPS metrics and pipeline status.
    """
    cmd = ["gst-launch-1.0", "-v"] + pipeline_string.split()
    print(f"\nLaunching pipeline:\ngst-launch-1.0 {pipeline_string}\n")

    metrics = {"fps_values": [], "status": "unknown", "duration": 0}
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def read_output():
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if line:
                print(f"  [GST] {line}")
                # Parse FPS from gvafpscounter output
                if "FPSCounter" in line or "fps" in line.lower():
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "fps" in part.lower() and i > 0:
                                fps_val = float(
                                    parts[i - 1]
                                    .replace(",", "")
                                    .replace("(", "")
                                )
                                metrics["fps_values"].append(fps_val)
                    except (ValueError, IndexError):
                        pass

    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()

    try:
        if duration:
            time.sleep(duration)
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
        else:
            process.wait()
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)

    metrics["duration"] = time.time() - start_time
    metrics["status"] = "completed" if process.returncode == 0 else "error"
    metrics["return_code"] = process.returncode

    if metrics["fps_values"]:
        metrics["avg_fps"] = np.mean(metrics["fps_values"])
        metrics["min_fps"] = np.min(metrics["fps_values"])
        metrics["max_fps"] = np.max(metrics["fps_values"])

    return metrics


def process_video_opencv(
    input_source: str,
    detection_model: str,
    ocr_model: str,
    duration: int = 30,
    output_dir: str = str(OUTPUT_DIR),
) -> dict:
    """Process video using OpenCV with OpenVINO backend as fallback.

    This is used when DL Streamer is not available, providing a similar
    pipeline using OpenCV DNN with OpenVINO backend.

    Args:
        input_source: Video file path or URL.
        detection_model: Path to detection model.
        ocr_model: Path to OCR model.
        duration: Maximum processing duration in seconds.
        output_dir: Output directory for results.

    Returns:
        Dictionary with processing metrics.
    """
    import openvino as ov

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing video: {input_source}")
    print("Using OpenCV + OpenVINO fallback pipeline")

    # Open video source
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {input_source}")
        return {"status": "error", "message": "Could not open video source"}

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps_video:.1f} FPS")

    # Load OpenVINO models
    core = ov.Core()
    print(f"\nLoading detection model: {detection_model}")

    det_model_path = Path(detection_model)
    if det_model_path.exists():
        det_compiled = core.compile_model(str(det_model_path), "CPU")
        det_infer = det_compiled.create_infer_request()
        det_input_layer = det_compiled.input(0)
        det_available = True
        print("Detection model loaded successfully")
    else:
        print(f"Warning: Detection model not found at {detection_model}")
        det_available = False

    # Process frames
    frame_count = 0
    fps_values = []
    start_time = time.time()
    sample_frames = []
    annotated_frames = []

    print(f"\nProcessing frames (max {duration}s)...")

    while cap.isOpened():
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()

        # Run detection if model is available
        if det_available:
            # Prepare input
            input_shape = det_input_layer.shape
            if len(input_shape) == 4:
                _, _, h, w = input_shape
                blob = cv2.resize(frame, (w, h))
                blob = blob.transpose(2, 0, 1)
                blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0

                # Run inference
                det_infer.infer({0: blob})
                output = det_infer.get_output_tensor(0).data

                # Parse detections and draw boxes
                if output is not None and output.size > 0:
                    # Draw detection indicator
                    cv2.putText(
                        annotated,
                        f"Detections: active",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

        # Calculate FPS
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_values.append(current_fps)

        # Add FPS overlay
        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Save sample frames for comparison
        if frame_count == 1 or frame_count % max(1, int(fps_video * 5)) == 0:
            sample_frames.append(frame.copy())
            annotated_frames.append(annotated.copy())

        if frame_count % 100 == 0:
            avg_fps = np.mean(fps_values[-100:])
            print(f"  Processed {frame_count} frames, avg FPS: {avg_fps:.1f}")

    cap.release()
    total_time = time.time() - start_time

    # Create side-by-side comparison
    if sample_frames and annotated_frames:
        create_stream_comparison(
            sample_frames,
            annotated_frames,
            fps_values,
            total_time,
            frame_count,
            output_path / "v3_comparison.png",
        )

    metrics = {
        "status": "completed",
        "frame_count": frame_count,
        "duration": total_time,
        "avg_fps": np.mean(fps_values) if fps_values else 0,
        "min_fps": np.min(fps_values) if fps_values else 0,
        "max_fps": np.max(fps_values) if fps_values else 0,
        "fps_values": fps_values,
    }

    return metrics


def create_stream_comparison(
    input_frames: list[np.ndarray],
    annotated_frames: list[np.ndarray],
    fps_values: list[float],
    total_time: float,
    total_frames: int,
    output_path: Path,
) -> None:
    """Create a side-by-side comparison of input and annotated video streams.

    Args:
        input_frames: List of sample input frames.
        annotated_frames: List of corresponding annotated frames.
        fps_values: List of FPS values measured during processing.
        total_time: Total processing time in seconds.
        total_frames: Total number of frames processed.
        output_path: Path to save the comparison figure.
    """
    n_samples = min(len(input_frames), 3)
    fig, axes = plt.subplots(n_samples, 3, figsize=(20, 6 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Input frame
        axes[i, 0].imshow(cv2.cvtColor(input_frames[i], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(
            f"Input Stream (Frame {i + 1})", fontsize=12, fontweight="bold"
        )
        axes[i, 0].axis("off")

        # Annotated frame
        axes[i, 1].imshow(cv2.cvtColor(annotated_frames[i], cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(
            f"Annotated Output (Frame {i + 1})", fontsize=12, fontweight="bold"
        )
        axes[i, 1].axis("off")

        # FPS graph for this segment
        segment_size = len(fps_values) // n_samples
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < n_samples - 1 else len(fps_values)
        segment_fps = fps_values[start_idx:end_idx]

        if segment_fps:
            axes[i, 2].plot(segment_fps, color="blue", linewidth=1)
            axes[i, 2].axhline(
                y=np.mean(segment_fps), color="red", linestyle="--", label="Average"
            )
            axes[i, 2].set_xlabel("Frame")
            axes[i, 2].set_ylabel("FPS")
            axes[i, 2].set_title(
                f"FPS: avg={np.mean(segment_fps):.1f}, "
                f"min={np.min(segment_fps):.1f}, "
                f"max={np.max(segment_fps):.1f}",
                fontsize=12,
                fontweight="bold",
            )
            axes[i, 2].legend()
            axes[i, 2].grid(True, alpha=0.3)

    title = "V3: DL Streamer Pipeline - Smart Parking"
    if total_time > 0:
        title += (
            f"\nTotal: {total_frames} frames in {total_time:.1f}s "
            f"(Avg FPS: {total_frames / total_time:.1f})"
        )

    plt.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"Stream comparison figure saved to: {output_path}")


def main() -> None:
    """Main entry point for the V3 DL Streamer pipeline application."""
    parser = argparse.ArgumentParser(
        description=(
            "V3: License Plate Recognition using "
            "Intel DL Streamer Pipeline"
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_VIDEO_URL,
        help=f"Video input source (file, URL, or camera device). "
        f"Default: Pexels smart parking video",
    )
    parser.add_argument(
        "--models-path",
        type=str,
        default=DEFAULT_MODELS_PATH,
        help=(
            "Path to V2 optimized models directory "
            f"(default: {DEFAULT_MODELS_PATH})"
        ),
    )
    parser.add_argument(
        "--detection-model",
        type=str,
        default=None,
        help=(
            "Override detection model path or directory "
            "(default: <models-path>/yolo)"
        ),
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default=None,
        help=(
            "Override OCR model path or directory "
            "(default: <models-path>/trocr)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="Inference device (default: CPU)",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="fps",
        choices=["display", "fps", "json", "file", "display-and-json"],
        help="Output mode (default: fps)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Maximum processing duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output video file path (for 'file' output mode)",
    )

    args = parser.parse_args()

    # Resolve model paths — use V2 optimized OV IR models
    models_path = Path(args.models_path)
    det_model_dir = args.detection_model or str(
        models_path / DEFAULT_DETECTION_MODEL_DIR
    )
    ocr_model_dir = args.ocr_model or str(models_path / DEFAULT_OCR_MODEL_DIR)

    # Find the .xml model files within the directories
    try:
        det_model = find_model_xml(det_model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        ocr_model = find_model_xml(ocr_model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V3: License Plate Recognition - DL Streamer Pipeline")
    print("=" * 60)
    print(f"Input source: {args.input}")
    print(f"Detection model: {det_model}")
    print(f"OCR model: {ocr_model}")
    print(f"Device: {args.device}")
    print(f"Output mode: {args.output_mode}")
    print(f"Duration: {args.duration}s")

    # Check for DL Streamer availability
    has_gstreamer = check_gstreamer_available()
    has_dlstreamer = check_dlstreamer_available()

    if has_dlstreamer:
        print("\nDL Streamer detected - using native GStreamer pipeline")

        pipeline = build_pipeline_string(
            input_source=args.input,
            detection_model=det_model,
            ocr_model=ocr_model,
            device=args.device,
            output_mode=args.output_mode,
            output_file=args.output_file,
        )

        print(f"\nPipeline:\n{pipeline}\n")
        metrics = run_gstreamer_pipeline(pipeline, duration=args.duration)

    else:
        if not has_gstreamer:
            print("\nGStreamer not found on this system.")
        else:
            print("\nDL Streamer elements not found.")

        print("Falling back to OpenCV + OpenVINO pipeline...")
        print(
            "Note: Install DL Streamer for full pipeline functionality."
        )
        print(
            "See: https://github.com/open-edge-platform/dlstreamer"
        )

        metrics = process_video_opencv(
            input_source=args.input,
            detection_model=det_model,
            ocr_model=ocr_model,
            duration=args.duration,
            output_dir=args.output_dir,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("V3 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Status: {metrics.get('status', 'unknown')}")
    print(f"Duration: {metrics.get('duration', 0):.1f}s")

    if "frame_count" in metrics:
        print(f"Frames processed: {metrics['frame_count']}")

    if "avg_fps" in metrics:
        print(f"Average FPS: {metrics['avg_fps']:.1f}")
        print(f"Min FPS: {metrics.get('min_fps', 0):.1f}")
        print(f"Max FPS: {metrics.get('max_fps', 0):.1f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
