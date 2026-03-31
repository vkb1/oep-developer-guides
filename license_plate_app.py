"""Integrated License Plate Detection & Recognition Application.

Three execution phases:
  Phase 1: Native YOLO .pt + PaddleOCR (non-optimized) on an image
  Phase 2: OpenVINO IR YOLO + PaddleOCR on an image (optimized)
  Phase 3: DL Streamer GStreamer pipeline on video

Usage:
    python license_plate_app.py --phase 1 --input image.jpg
    python license_plate_app.py --phase 2 --input image.jpg --device CPU
    python license_plate_app.py --phase 3 --input video.mp4 --device CPU
    python license_plate_app.py --phase all --input image.jpg --video video.mp4
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# --- Non-optimized (native .pt) base path  (Phase 1) ---
YOLO_NATIVE_BASE = MODELS_DIR / "morsetechlab_yolov11-license-plate-detection"

# --- OpenVINO-optimized base path  (Phase 2 / 3) ---
YOLO_OV_BASE = (
    MODELS_DIR
    / "morsetechlab_yolov11-license-plate-detection"
    / "models_ov"
)

PPOCR_MODEL_DIR = MODELS_DIR / "PaddlePaddle_PP-OCRv4_server_rec"

VALID_MODEL_SIZES = ["v1n", "v1s", "v1m", "v1l", "v1x"]
DEFAULT_MODEL_SIZE = "v1m"


def _yolo_pt_path(size: str = DEFAULT_MODEL_SIZE) -> Path:
    """Return path to the native YOLO .pt model (Phase 1)."""
    p = YOLO_NATIVE_BASE / f"license-plate-finetune-{size}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"YOLO .pt model not found at {p}.\n"
            f"Available sizes: {VALID_MODEL_SIZES}"
        )
    return p


def _yolo_ov_path(size: str = DEFAULT_MODEL_SIZE) -> Path:
    """Return path to the OpenVINO IR YOLO model directory (Phase 2/3).

    Ultralytics expects the *directory* containing the .xml, .bin, and
    metadata.yaml files, not the .xml file itself.
    """
    stem = f"license-plate-finetune-{size}"
    model_dir = YOLO_OV_BASE / f"{stem}_openvino_model"
    xml_file = model_dir / f"{stem}.xml"
    if not xml_file.exists():
        raise FileNotFoundError(
            f"YOLO OpenVINO model not found at {xml_file}.\n"
            f"Available sizes: {VALID_MODEL_SIZES}"
        )
    return model_dir


def _ppocr_ov_xml() -> Path:
    return PPOCR_MODEL_DIR / "pp_ocrv4_server_rec.xml"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _detect(model, image, confidence: float, iou: float = 0.45):
    """Run YOLO detection; return (detections_list, elapsed_ms)."""
    start = time.perf_counter()
    results = model.predict(image, conf=confidence, iou=iou, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000

    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dets.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(box.conf[0].cpu().numpy()),
                "label": r.names.get(int(box.cls[0].cpu().numpy()), "plate"),
            })
    return dets, elapsed


def _crop_plates(image, dets, pad=5):
    h, w = image.shape[:2]
    crops = []
    for d in dets:
        x1, y1, x2, y2 = (int(c) for c in d["bbox"])
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        c = image[y1:y2, x1:x2]
        if c.size > 0:
            crops.append(c)
    return crops


def _ocr(crops, ocr_engine=None):
    """PaddleOCR on crops; return (texts, elapsed_ms).

    Uses PaddleOCR v3.x ``predict()`` API which returns an iterator of
    dicts with ``rec_texts`` and ``rec_scores`` keys.

    Pass a pre-created ``ocr_engine`` to avoid re-instantiating the model
    on every call (e.g. inside a video frame loop).
    """
    from paddleocr import PaddleOCR

    if ocr_engine is None:
        ocr_engine = PaddleOCR(
            lang="en",
            rec_model_dir=str(PPOCR_MODEL_DIR),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        # Warmup: PaddleOCR lazy-loads its models on the first predict() call;
        # run a dummy pass so that load time is excluded from OCR timing.
        _dummy = np.zeros((32, 100, 3), dtype=np.uint8)
        for _ in ocr_engine.predict(_dummy):
            break

    ocr = ocr_engine
    texts = []
    start = time.perf_counter()
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plate = ""
        for res in ocr.predict(rgb):
            rec_texts = res.get("rec_texts", [])
            plate = " ".join(rec_texts)
            break  # single image, take first result
        texts.append(plate.strip() if plate.strip() else "[unreadable]")
    elapsed = (time.perf_counter() - start) * 1000
    return texts, elapsed


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw(image, dets, color=(0, 255, 0), thickness=2):
    out = image.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(c) for c in d["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        txt = f"{d['label']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return out


# ---------------------------------------------------------------------------
# Visualization -- 3-panel figure
# ---------------------------------------------------------------------------
def _make_figure(original, annotated, dets, texts, times, title, out_path):
    """Create and save a 3-panel comparison figure."""
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=100)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(ann_rgb)
    axes[1].set_title(f"Detection ({times['det_ms']:.1f} ms)", fontsize=12)
    axes[1].axis("off")

    lines = [
        "Inference Results",
        "=" * 34,
        "",
        f"Plates detected : {len(dets)}",
        f"Detection time  : {times['det_ms']:.1f} ms",
        f"OCR time        : {times['ocr_ms']:.1f} ms",
        f"Total time      : {times['total_ms']:.1f} ms",
        "",
        "-" * 34,
        "Recognized Plates:",
        "-" * 34,
    ]
    if texts:
        for i, t in enumerate(texts, 1):
            conf = dets[i - 1]["confidence"] if i <= len(dets) else 0
            lines += [f"  #{i}  {t}", f"       conf: {conf:.2f}"]
    else:
        lines.append("  No plates detected")
    lines += ["", "-" * 34]
    for d in dets:
        b = d["bbox"]
        lines.append(f"  Box: ({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})")

    axes[2].text(0.5, 0.5, "\n".join(lines),
                 transform=axes[2].transAxes, fontsize=11,
                 va="center", ha="center", fontfamily="monospace",
                 bbox={"boxstyle": "round,pad=0.6",
                       "facecolor": "lightyellow", "alpha": 0.9})
    axes[2].set_title("Inference Numbers", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=100)
    plt.close(fig)
    logger.info("Figure saved -> %s", out_path)
    return out_path


def _make_video_figure(inp_frame, ann_frame, avg_fps, frame_no,
                       unique_texts, times, out_path):
    """Create a 3-panel figure for a representative video frame."""
    inp_rgb = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)
    ann_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=100)
    fig.suptitle("Phase 3 -- DL Streamer / Video Pipeline", fontsize=15,
                 fontweight="bold")

    axes[0].imshow(inp_rgb)
    axes[0].set_title(f"Input Frame #{frame_no}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(ann_rgb)
    axes[1].set_title("Annotated Frame", fontsize=12)
    axes[1].axis("off")

    lines = [
        "Video Inference Results",
        "=" * 34,
        "",
        f"Avg FPS         : {avg_fps:.1f}",
        f"Avg det/frame   : {times['avg_det_ms']:.1f} ms",
        f"Total OCR time  : {times['total_ocr_ms']:.1f} ms",
        f"Frames processed: {times['frames']}",
        "",
        "-" * 34,
        "Unique Plates Seen:",
        "-" * 34,
    ]
    if unique_texts:
        for i, t in enumerate(unique_texts, 1):
            lines.append(f"  #{i}  {t}")
    else:
        lines.append("  None")

    axes[2].text(0.5, 0.5, "\n".join(lines),
                 transform=axes[2].transAxes, fontsize=11,
                 va="center", ha="center", fontfamily="monospace",
                 bbox={"boxstyle": "round,pad=0.6",
                       "facecolor": "lightyellow", "alpha": 0.9})
    axes[2].set_title("Inference Numbers", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=100)
    plt.close(fig)
    logger.info("Video figure saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Phase 1: Native .pt YOLO + PaddleOCR (non-optimized)
# ---------------------------------------------------------------------------
def run_phase1(input_image, confidence=0.5, model_size=DEFAULT_MODEL_SIZE,
               output=None):
    """Phase 1 -- native YOLO .pt model + PaddleOCR on a single image."""
    from ultralytics import YOLO

    pt_path = _yolo_pt_path(model_size)
    logger.info("[Phase 1] YOLO .pt  : %s  (YOLO_NATIVE_BASE)", pt_path)
    logger.info("[Phase 1] OCR dir   : %s", PPOCR_MODEL_DIR)

    img = cv2.imread(str(input_image))
    if img is None:
        raise ValueError(f"Cannot read image: {input_image}")
    logger.info("[Phase 1] Image %s (%dx%d)", Path(input_image).name,
                img.shape[1], img.shape[0])

    total_start = time.perf_counter()

    model = YOLO(str(pt_path))
    dets, det_ms = _detect(model, img, confidence)
    annotated = _draw(img, dets)
    crops = _crop_plates(img, dets)
    texts, ocr_ms = _ocr(crops)
    total_ms = (time.perf_counter() - total_start) * 1000

    times = {"det_ms": det_ms, "ocr_ms": ocr_ms, "total_ms": total_ms}

    out_dir = OUTPUT_DIR / "phase1"
    stem = Path(input_image).stem
    out_path = Path(output) if output else out_dir / f"{stem}_phase1.png"

    _make_figure(img, annotated, dets, texts, times,
                 "Phase 1 -- Native YOLO + PaddleOCR (non-optimized)", out_path)

    _print_summary("Phase 1", dets, texts, times, out_path)
    return {"detections": dets, "texts": texts, "times": times,
            "output": str(out_path)}


# ---------------------------------------------------------------------------
# Phase 2: OpenVINO-optimized YOLO + PaddleOCR
# ---------------------------------------------------------------------------
def run_phase2(input_image, confidence=0.5, model_size=DEFAULT_MODEL_SIZE,
               device="CPU", output=None):
    """Phase 2 -- OpenVINO IR YOLO model + PaddleOCR on a single image."""
    from ultralytics import YOLO

    ov_path = _yolo_ov_path(model_size)
    logger.info("[Phase 2] YOLO OV   : %s  (YOLO_OV_BASE)", ov_path)
    logger.info("[Phase 2] OCR dir   : %s", PPOCR_MODEL_DIR)

    img = cv2.imread(str(input_image))
    if img is None:
        raise ValueError(f"Cannot read image: {input_image}")
    logger.info("[Phase 2] Image %s (%dx%d)", Path(input_image).name,
                img.shape[1], img.shape[0])

    model = YOLO(str(ov_path), task="detect")
    # Warmup: OpenVINO JIT-compiles the model on the first predict() call;
    # without this, that compilation time is counted as detection latency.
    logger.info("[Phase 2] Warming up OpenVINO model (first-run compilation)...")
    model.predict(img, conf=confidence, iou=0.45, verbose=False)

    total_start = time.perf_counter()
    dets, det_ms = _detect(model, img, confidence)
    annotated = _draw(img, dets, color=(255, 165, 0))
    crops = _crop_plates(img, dets)
    texts, ocr_ms = _ocr(crops)
    total_ms = (time.perf_counter() - total_start) * 1000

    times = {"det_ms": det_ms, "ocr_ms": ocr_ms, "total_ms": total_ms}

    out_dir = OUTPUT_DIR / "phase2"
    stem = Path(input_image).stem
    out_path = Path(output) if output else out_dir / f"{stem}_phase2.png"

    _make_figure(img, annotated, dets, texts, times,
                 f"Phase 2 -- OpenVINO Optimized ({device})", out_path)

    _print_summary("Phase 2", dets, texts, times, out_path)
    return {"detections": dets, "texts": texts, "times": times,
            "output": str(out_path)}


# ---------------------------------------------------------------------------
# Phase 3: DL Streamer pipeline (video)
# ---------------------------------------------------------------------------
OCR_FRAME_INTERVAL = 10
SAMPLE_FRAME_INDEX = 30


def _gstreamer_available():
    try:
        r = subprocess.run(["gst-launch-1.0", "--version"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_dlstreamer_pipeline(input_video, detection_model, ocr_model,
                               device, output_mode="fps"):
    """Build a DL Streamer pipeline string adapted from the
    open-edge-platform/dlstreamer license_plate_recognition sample."""

    # Source element
    if input_video.startswith("/dev/video"):
        source = f"v4l2src device={input_video}"
    elif "://" in input_video:
        source = f"urisourcebin buffer-size=4096 uri={input_video}"
    else:
        source = f"filesrc location={input_video}"

    # Decode / pre-process
    if device == "GPU":
        decode = "decodebin3 ! vapostproc ! video/x-raw(memory:VAMemory)"
        preproc = "pre-process-backend=va-surface-sharing"
    else:
        decode = "decodebin3"
        preproc = "pre-process-backend=opencv"

    # Detect + classify
    detect = f"gvadetect model={detection_model} device={device} {preproc}"
    classify = ""
    if ocr_model:
        classify = (f"! queue ! videoconvert ! "
                    f"gvaclassify model={ocr_model} device={device} "
                    f"{preproc}")

    # Sink
    if output_mode == "display":
        sink = ("vapostproc ! gvawatermark ! videoconvert ! "
                "gvafpscounter ! autovideosink")
    elif output_mode == "json":
        sink = ("gvametaconvert ! gvametapublish file-format=json-lines "
                "file-path=output.json ! fakesink async=false")
    else:  # fps
        sink = "gvafpscounter ! fakesink async=false"

    parts = [
        f"gst-launch-1.0 {source}",
        f"! {decode}",
        f"! queue ! {detect}",
    ]
    if classify:
        parts.append(classify)
    parts.append(f"! queue ! {sink}")

    return " ".join(parts)


def _run_video_opencv(input_video, model, confidence, max_frames, out_dir):
    """Fallback: process video frame-by-frame with OpenCV + YOLO + OCR."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video %s %dx%d %.1f FPS %d frames",
                input_video, w, h, fps, total_frames)

    # Create and warm up OCR engine once for the entire video.
    from paddleocr import PaddleOCR
    logger.info("[Phase 3] Initialising PaddleOCR engine (once for all frames)...")
    _ocr_engine = PaddleOCR(
        lang="en",
        rec_model_dir=str(PPOCR_MODEL_DIR),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    _dummy = np.zeros((32, 100, 3), dtype=np.uint8)
    for _ in _ocr_engine.predict(_dummy):
        break

    out_vid = out_dir / "annotated_output.mp4"
    writer = cv2.VideoWriter(str(out_vid),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    limit = max_frames if max_frames > 0 else total_frames
    count = 0
    total_det = 0.0
    total_ocr = 0.0
    all_texts = []
    fps_vals = []
    sample_in = sample_out = None

    while cap.isOpened() and count < limit:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        dets, dt = _detect(model, frame, confidence)
        total_det += dt
        annotated = _draw(frame, dets)

        if dets and count % OCR_FRAME_INTERVAL == 0:
            crops = _crop_plates(frame, dets)
            texts, ot = _ocr(crops, ocr_engine=_ocr_engine)
            total_ocr += ot
            all_texts.extend(texts)

        frame_ms = (time.perf_counter() - t0) * 1000
        cur_fps = 1000.0 / frame_ms if frame_ms > 0 else 0
        fps_vals.append(cur_fps)

        cv2.putText(annotated, f"FPS: {cur_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(annotated)

        if count == min(SAMPLE_FRAME_INDEX, limit - 1):
            sample_in = frame.copy()
            sample_out = annotated.copy()

        count += 1
        if count % 50 == 0:
            logger.info("Processed %d/%d frames", count, limit)

    cap.release()
    writer.release()

    avg_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0
    unique = sorted(set(all_texts) - {"[unreadable]"})

    times_info = {
        "avg_det_ms": total_det / max(count, 1),
        "total_ocr_ms": total_ocr,
        "frames": count,
    }

    if sample_in is not None and sample_out is not None:
        fig_path = out_dir / "video_comparison.png"
        _make_video_figure(sample_in, sample_out, avg_fps,
                           min(SAMPLE_FRAME_INDEX, count - 1),
                           unique, times_info, fig_path)

    logger.info("Video done: %d frames, avg FPS %.1f", count, avg_fps)
    return {"frames": count, "avg_fps": avg_fps, "unique_plates": unique,
            "output_video": str(out_vid),
            "comparison": str(out_dir / "video_comparison.png")}


def run_phase3(input_video, confidence=0.5, model_size=DEFAULT_MODEL_SIZE,
               device="CPU", output_mode="fps", max_frames=0, output=None):
    """Phase 3 -- DL Streamer pipeline (or OpenCV fallback) on video.

    Uses YOLO_OV_BASE for the detection model (OpenVINO IR)."""
    from ultralytics import YOLO

    ov_det = _yolo_ov_path(model_size)
    # gvadetect requires the explicit .xml file; Ultralytics uses the directory
    ov_det_xml = ov_det / f"license-plate-finetune-{model_size}.xml"
    ov_ocr = _ppocr_ov_xml() if _ppocr_ov_xml().exists() else None
    out_dir = Path(output) if output else OUTPUT_DIR / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(input_video).exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    use_dls = _gstreamer_available()

    if use_dls:
        logger.info("[Phase 3] DL Streamer available -- building pipeline")
        pipeline = _build_dlstreamer_pipeline(
            input_video, str(ov_det_xml),
            str(ov_ocr) if ov_ocr else None,
            device, output_mode)
        logger.info("Pipeline:\n  %s", pipeline)

        dls_setup = "source /opt/intel/dlstreamer/scripts/setup_dls_env.sh"
        shell_cmd = f"{dls_setup} && {pipeline}"
        start = time.perf_counter()
        result = subprocess.run(
            shell_cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=600)
        duration = time.perf_counter() - start

        if result.returncode != 0:
            logger.warning("DL Streamer failed (rc=%d), falling back to OpenCV",
                           result.returncode)
            use_dls = False
        else:
            logger.info("DL Streamer finished in %.1f s", duration)
            stdout = result.stdout
            print(stdout[-2000:] if len(stdout) > 2000 else stdout)
            return {"pipeline": pipeline, "duration_s": duration,
                    "stdout": stdout, "stderr": result.stderr}

    if not use_dls:
        logger.info("[Phase 3] Using OpenCV fallback pipeline")
        model = YOLO(str(ov_det))
        # Warmup: trigger OpenVINO JIT compilation before processing begins
        _warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(_warmup_frame, conf=confidence, iou=0.45, verbose=False)
        return _run_video_opencv(input_video, model, confidence,
                                 max_frames, out_dir)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def _print_summary(phase, dets, texts, times, out_path):
    sep = "=" * 44
    print(f"\n{sep}")
    print(f"  {phase} -- License Plate Results")
    print(sep)
    print(f"  Plates found  : {len(dets)}")
    print(f"  Detection     : {times['det_ms']:.1f} ms")
    print(f"  OCR           : {times['ocr_ms']:.1f} ms")
    print(f"  Total         : {times['total_ms']:.1f} ms")
    if texts:
        print("  " + "-" * 40)
        for i, t in enumerate(texts, 1):
            print(f"  Plate #{i}: {t}")
    print(sep)
    print(f"  Output: {out_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="License Plate Detection & Recognition -- 3 Phases",
    )
    parser.add_argument(
        "--phase", required=True,
        choices=["1", "2", "3", "all"],
        help="Phase to run: 1 (native), 2 (OpenVINO), 3 (DL Streamer), all",
    )
    parser.add_argument("--input", help="Path to input image (phases 1/2)")
    parser.add_argument("--video", help="Path to input video (phase 3)")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--model-size", choices=VALID_MODEL_SIZES,
                        default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--device", default="CPU",
                        help="OpenVINO device (CPU, GPU, AUTO)")
    parser.add_argument("--output-mode", default="fps",
                        choices=["display", "fps", "json"],
                        help="DL Streamer output mode (phase 3)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max video frames to process (0=all)")
    parser.add_argument("--output", default=None,
                        help="Custom output path")
    args = parser.parse_args()

    phases = [args.phase] if args.phase != "all" else ["1", "2", "3"]

    for phase in phases:
        try:
            if phase in ("1", "2"):
                if not args.input:
                    parser.error(f"--input required for phase {phase}")
                if phase == "1":
                    run_phase1(args.input, args.confidence,
                               args.model_size, args.output)
                else:
                    run_phase2(args.input, args.confidence,
                               args.model_size, args.device, args.output)
            elif phase == "3":
                video = args.video or args.input
                if not video:
                    parser.error("--video (or --input) required for phase 3")
                run_phase3(video, args.confidence, args.model_size,
                           args.device, args.output_mode,
                           args.max_frames, args.output)
        except Exception as e:
            logger.error("Phase %s failed: %s", phase, e)
            if len(phases) == 1:
                sys.exit(1)


if __name__ == "__main__":
    main()
