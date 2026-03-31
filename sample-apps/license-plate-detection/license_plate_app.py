#!/usr/bin/env python3
"""License Plate Detection & OCR — three-phase application.

Phase 1  –  image  │ YOLO .pt (native PyTorch)   │ PaddleOCR   │ CPU
Phase 2  –  image  │ YOLO OpenVINO IR (directory) │ PaddleOCR   │ OpenVINO
Phase 3  –  video  │ YOLO OpenVINO IR (.xml)      │ gvaclassify │ DL Streamer
           (falls back to OpenCV + PaddleOCR when DL Streamer is unavailable)

Usage
-----
    # Run all phases with defaults
    python license_plate_app.py --image car.jpg --video traffic.mp4

    # Phase 1 only, custom model size
    python license_plate_app.py --image car.jpg --phase 1 --model-size v1s

    # Phase 3 only (DL Streamer or OpenCV fallback)
    python license_plate_app.py --video traffic.mp4 --phase 3
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless – no display required
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Path constants — never mix native vs. OpenVINO base directories
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Phase 1: native .pt models live here
YOLO_NATIVE_BASE = MODELS_DIR / "morsetechlab_yolov11-license-plate-detection"
# Phase 2 / 3: OpenVINO IR models live here
YOLO_OV_BASE = (
    MODELS_DIR / "morsetechlab_yolov11-license-plate-detection" / "models_ov"
)
# PP-OCRv4 recognition model for PaddleOCR
PPOCR_MODEL_DIR = MODELS_DIR / "PaddlePaddle_PP-OCRv4_server_rec"

# ── Tunables ──────────────────────────────────────────────────────────────
VALID_SIZES = ("v1n", "v1s", "v1m", "v1l", "v1x")
DEFAULT_SIZE = "v1m"
DEFAULT_CONFIDENCE = 0.25
OCR_FRAME_INTERVAL = 10  # Phase 3 fallback: run OCR every N frames
DLS_ENV_SCRIPT = "/opt/intel/dlstreamer/scripts/setup_dls_env.sh"


# ═══════════════════════════════════════════════════════════════════════════
# Model-path helpers
# ═══════════════════════════════════════════════════════════════════════════
def _yolo_pt_path(model_size: str) -> Path:
    """Return the .pt path for a given model size (Phase 1)."""
    return YOLO_NATIVE_BASE / f"license-plate-finetune-{model_size}.pt"


def _yolo_ov_dir(model_size: str) -> Path:
    """Return the OpenVINO IR *directory* (Phase 2 – pass to YOLO())."""
    return YOLO_OV_BASE / f"license-plate-finetune-{model_size}_openvino_model"


def _yolo_ov_xml(model_size: str) -> Path:
    """Return the OpenVINO IR .xml *file* (Phase 3 – pass to gvadetect)."""
    return _yolo_ov_dir(model_size) / f"license-plate-finetune-{model_size}.xml"


# ═══════════════════════════════════════════════════════════════════════════
# PaddleOCR initialisation (shared by Phase 1, 2, and 3 fallback)
# ═══════════════════════════════════════════════════════════════════════════
def _create_ocr_engine() -> Any:
    """Instantiate and warm up a PaddleOCR v3.x engine.

    Critical settings:
      • rec_model_dir  → forces local PP-OCRv4 (prevents auto-downloading v5)
      • Three document-oriented models disabled (not useful for plate crops)
      • enable_mkldnn=False  → avoids OneDNN attribute errors in PaddlePaddle
    """
    from paddleocr import PaddleOCR  # noqa: E402

    engine = PaddleOCR(
        lang="en",
        rec_model_dir=str(PPOCR_MODEL_DIR),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )

    # Warmup — 32×100 is the minimum size PaddleOCR accepts for recognition
    _dummy = np.zeros((32, 100, 3), dtype=np.uint8)
    for _ in engine.predict(_dummy):
        break
    log.info("PaddleOCR engine warmed up.")
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# Detection helper (Ultralytics – works for both .pt and OV directory)
# ═══════════════════════════════════════════════════════════════════════════
def _detect(
    model: Any,
    image: np.ndarray,
    confidence: float,
) -> tuple[list[dict], float]:
    """Run YOLO detection and return (detections, elapsed_ms).

    Each detection dict: {x1, y1, x2, y2, conf, cls}
    """
    t0 = time.perf_counter()
    results = model.predict(image, conf=confidence, iou=0.45, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections: list[dict] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append(
                {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                }
            )
    return detections, elapsed_ms


# ═══════════════════════════════════════════════════════════════════════════
# OCR helper (PaddleOCR v3.x — always uses predict(), never ocr.ocr())
# ═══════════════════════════════════════════════════════════════════════════
def _ocr(
    crops: list[np.ndarray],
    ocr_engine: Any,
) -> tuple[list[str], float]:
    """Run OCR on a list of plate crops. Returns (texts, elapsed_ms)."""
    texts: list[str] = []
    t0 = time.perf_counter()
    for crop in crops:
        if crop.size == 0:
            texts.append("")
            continue
        # PaddleOCR expects RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = list(ocr_engine.predict(rgb))
        if results:
            rec_texts = results[0].get("rec_texts", [])
            plate_text = " ".join(rec_texts).strip()
            texts.append(plate_text)
        else:
            texts.append("")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return texts, elapsed_ms


# ═══════════════════════════════════════════════════════════════════════════
# Crop extraction
# ═══════════════════════════════════════════════════════════════════════════
def _extract_crops(
    image: np.ndarray,
    detections: list[dict],
    padding: int = 4,
) -> list[np.ndarray]:
    """Extract bounding-box crops from the image with optional padding."""
    h, w = image.shape[:2]
    crops: list[np.ndarray] = []
    for d in detections:
        x1 = max(0, d["x1"] - padding)
        y1 = max(0, d["y1"] - padding)
        x2 = min(w, d["x2"] + padding)
        y2 = min(h, d["y2"] + padding)
        crops.append(image[y1:y2, x1:x2])
    return crops


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation – 3-panel figure
# ═══════════════════════════════════════════════════════════════════════════
def _annotate_image(
    image: np.ndarray,
    detections: list[dict],
    texts: list[str],
) -> np.ndarray:
    """Draw bounding boxes and recognised text onto a copy of the image."""
    annotated = image.copy()
    for det, txt in zip(detections, texts):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["conf"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{txt}  {conf:.2f}" if txt else f"{conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return annotated


def _build_text_panel(
    det_ms: float,
    ocr_ms: float,
    total_ms: float,
    detections: list[dict],
    texts: list[str],
    extra_lines: list[str] | None = None,
) -> str:
    """Compose the multi-line string shown in the third panel."""
    lines = [
        "─── Inference Metrics ───",
        f"Detection:   {det_ms:8.1f} ms",
        f"OCR:         {ocr_ms:8.1f} ms",
        f"Total:       {total_ms:8.1f} ms",
        f"Plates:      {len(detections):>5d}",
        "",
        "─── Recognised Text ─────",
    ]
    for i, (det, txt) in enumerate(zip(detections, texts)):
        lines.append(
            f"  [{i}] '{txt}'  conf={det['conf']:.2f}"
        )
        lines.append(
            f"      bbox=({det['x1']}, {det['y1']}, {det['x2']}, {det['y2']})"
        )
    if not detections:
        lines.append("  (no plates detected)")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    return "\n".join(lines)


def _save_three_panel(
    original: np.ndarray,
    annotated: np.ndarray,
    text_content: str,
    output_path: Path,
    title: str = "",
) -> None:
    """Save a 3-panel side-by-side figure (original │ annotated │ text)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1 – original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panel 2 – annotated
    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detections + OCR")
    axes[1].axis("off")

    # Panel 3 – text metrics
    axes[2].axis("off")
    axes[2].set_title("Metrics")
    axes[2].text(
        0.05,
        0.95,
        text_content,
        transform=axes[2].transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "alpha": 0.8},
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    log.info("Saved visualisation → %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Native YOLO .pt + PaddleOCR (CPU / PyTorch)
# ═══════════════════════════════════════════════════════════════════════════
def run_phase1(
    image_path: str,
    model_size: str = DEFAULT_SIZE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Phase 1: native .pt detection + PaddleOCR on a single image."""
    from ultralytics import YOLO

    log.info("═══ Phase 1 ═══  (native .pt + PaddleOCR)")
    pt_path = _yolo_pt_path(model_size)
    if not pt_path.exists():
        log.error("Model not found: %s — run download_and_convert_openvino.py first.", pt_path)
        return

    image = cv2.imread(image_path)
    if image is None:
        log.error("Cannot read image: %s", image_path)
        return

    model = YOLO(str(pt_path))
    ocr_engine = _create_ocr_engine()

    # Warmup (exclude JIT / first-run overhead from timing)
    model.predict(image, conf=confidence, iou=0.45, verbose=False)

    # ── Timed run ────────────────────────────────────────────────────────
    total_start = time.perf_counter()
    detections, det_ms = _detect(model, image, confidence)
    crops = _extract_crops(image, detections)
    texts, ocr_ms = _ocr(crops, ocr_engine)
    total_ms = (time.perf_counter() - total_start) * 1000

    # ── Visualise ────────────────────────────────────────────────────────
    annotated = _annotate_image(image, detections, texts)
    text_panel = _build_text_panel(det_ms, ocr_ms, total_ms, detections, texts)
    out_path = OUTPUT_DIR / "phase1" / f"result_{Path(image_path).stem}.png"
    _save_three_panel(image, annotated, text_panel, out_path, title="Phase 1 — Native .pt + PaddleOCR")

    log.info("Phase 1 complete  det=%.1f ms  ocr=%.1f ms  total=%.1f ms  plates=%d",
             det_ms, ocr_ms, total_ms, len(detections))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — OpenVINO IR (directory) + PaddleOCR
# ═══════════════════════════════════════════════════════════════════════════
def run_phase2(
    image_path: str,
    model_size: str = DEFAULT_SIZE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Phase 2: OpenVINO IR detection + PaddleOCR on a single image."""
    from ultralytics import YOLO

    log.info("═══ Phase 2 ═══  (OpenVINO IR + PaddleOCR)")
    ov_dir = _yolo_ov_dir(model_size)
    if not ov_dir.exists():
        log.error("OpenVINO model dir not found: %s — run download_and_convert_openvino.py first.", ov_dir)
        return

    image = cv2.imread(image_path)
    if image is None:
        log.error("Cannot read image: %s", image_path)
        return

    # Pass the *directory* to YOLO — never the .xml file
    model = YOLO(str(ov_dir), task="detect")
    ocr_engine = _create_ocr_engine()

    # Warmup (OpenVINO first-run JIT can add 1-5 s)
    model.predict(image, conf=confidence, iou=0.45, verbose=False)

    # ── Timed run ────────────────────────────────────────────────────────
    total_start = time.perf_counter()
    detections, det_ms = _detect(model, image, confidence)
    crops = _extract_crops(image, detections)
    texts, ocr_ms = _ocr(crops, ocr_engine)
    total_ms = (time.perf_counter() - total_start) * 1000

    # ── Visualise ────────────────────────────────────────────────────────
    annotated = _annotate_image(image, detections, texts)
    text_panel = _build_text_panel(det_ms, ocr_ms, total_ms, detections, texts)
    out_path = OUTPUT_DIR / "phase2" / f"result_{Path(image_path).stem}.png"
    _save_three_panel(image, annotated, text_panel, out_path, title="Phase 2 — OpenVINO IR + PaddleOCR")

    log.info("Phase 2 complete  det=%.1f ms  ocr=%.1f ms  total=%.1f ms  plates=%d",
             det_ms, ocr_ms, total_ms, len(detections))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — DL Streamer (gvadetect / gvaclassify) or OpenCV fallback
# ═══════════════════════════════════════════════════════════════════════════
def _dlstreamer_available() -> bool:
    """Check whether gst-launch-1.0 and DL Streamer env script exist."""
    dls_script = Path(DLS_ENV_SCRIPT)
    if not dls_script.exists():
        return False
    try:
        result = subprocess.run(
            f"source {shlex.quote(str(dls_script))} && which gst-launch-1.0",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _run_dlstreamer_pipeline(
    video_path: str,
    model_size: str,
    confidence: float,
) -> tuple[bool, str]:
    """Build and execute a DL Streamer GStreamer pipeline.

    Returns (success, stderr_or_stdout).
    """
    ov_xml = _yolo_ov_xml(model_size)
    if not ov_xml.exists():
        return False, f"OpenVINO .xml not found: {ov_xml}"

    ppocr_xml = PPOCR_MODEL_DIR / "pp_ocrv4_server_rec.xml"
    ocr_model_arg = ""
    if ppocr_xml.exists():
        ocr_model_arg = (
            f" ! gvaclassify model={shlex.quote(str(ppocr_xml))} model-proc='' device=CPU"
        )

    phase3_dir = OUTPUT_DIR / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    pipeline = (
        f"gst-launch-1.0 "
        f"filesrc location={shlex.quote(str(video_path))} ! decodebin ! videoconvert ! "
        f"video/x-raw,format=BGRx ! "
        f"gvadetect model={shlex.quote(str(ov_xml))} device=CPU threshold={confidence} ! "
        f"queue"
        f"{ocr_model_arg} ! "
        f"gvafpscounter ! fakesink"
    )

    dls_setup = f"source {shlex.quote(DLS_ENV_SCRIPT)}"
    shell_cmd = f"{dls_setup} && {pipeline}"
    log.info("DL Streamer pipeline:\n  %s", pipeline)

    try:
        result = subprocess.run(
            shell_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout + "\n" + result.stderr
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Pipeline timed out after 600 s"


def _run_opencv_fallback(
    video_path: str,
    model_size: str,
    confidence: float,
) -> None:
    """Phase 3 OpenCV fallback: frame-by-frame YOLO + OCR every N frames."""
    from ultralytics import YOLO

    log.info("Phase 3 fallback: OpenCV + YOLO OV + PaddleOCR")

    ov_dir = _yolo_ov_dir(model_size)
    if not ov_dir.exists():
        log.error("OpenVINO model dir not found: %s", ov_dir)
        return

    model = YOLO(str(ov_dir), task="detect")
    ocr_engine = _create_ocr_engine()

    # Warmup with a 640×640 black frame
    _warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(_warmup_frame, conf=confidence, iou=0.45, verbose=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info("Video: %d frames @ %.1f FPS", total_frames, fps)

    frame_idx = 0
    all_det_ms: list[float] = []
    all_ocr_ms: list[float] = []
    last_texts: list[str] = []
    last_detections: list[dict] = []
    sample_original: np.ndarray | None = None
    sample_annotated: np.ndarray | None = None

    total_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, det_ms = _detect(model, frame, confidence)
        all_det_ms.append(det_ms)

        # Run OCR only every OCR_FRAME_INTERVAL frames *if* there are detections
        if detections and frame_idx % OCR_FRAME_INTERVAL == 0:
            crops = _extract_crops(frame, detections)
            texts, ocr_ms = _ocr(crops, ocr_engine)
            all_ocr_ms.append(ocr_ms)
            last_texts = texts
            last_detections = detections

            # Keep the first frame with detections as the sample
            if sample_original is None:
                sample_original = frame.copy()
                sample_annotated = _annotate_image(frame, detections, texts)
        elif detections:
            # Use previous OCR results for annotating
            last_detections = detections
            # Ensure texts list matches detections length for zip safety
            if len(last_texts) != len(detections):
                last_texts = [""] * len(detections)

        frame_idx += 1

    total_ms = (time.perf_counter() - total_start) * 1000
    cap.release()

    avg_det = np.mean(all_det_ms) if all_det_ms else 0.0
    avg_ocr = np.mean(all_ocr_ms) if all_ocr_ms else 0.0

    log.info(
        "Processed %d frames  avg_det=%.1f ms  avg_ocr=%.1f ms  total=%.1f ms",
        frame_idx, avg_det, avg_ocr, total_ms,
    )

    # ── Visualise using the sample frame ─────────────────────────────────
    if sample_original is not None and sample_annotated is not None:
        text_panel = _build_text_panel(
            avg_det,
            avg_ocr,
            total_ms,
            last_detections,
            last_texts,
            extra_lines=[
                f"Frames processed:  {frame_idx}",
                f"Video FPS:         {fps:.1f}",
                f"OCR interval:      every {OCR_FRAME_INTERVAL} frames",
            ],
        )
        out_path = OUTPUT_DIR / "phase3" / f"result_{Path(video_path).stem}.png"
        _save_three_panel(
            sample_original,
            sample_annotated,
            text_panel,
            out_path,
            title="Phase 3 — OpenCV fallback (YOLO OV + PaddleOCR)",
        )
    else:
        log.warning("No detections found in video — no visualisation saved.")


def run_phase3(
    video_path: str,
    model_size: str = DEFAULT_SIZE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Phase 3: DL Streamer pipeline (or OpenCV fallback) on video."""
    log.info("═══ Phase 3 ═══  (DL Streamer / OpenCV fallback on video)")

    if _dlstreamer_available():
        log.info("DL Streamer detected — running GStreamer pipeline.")
        success, output = _run_dlstreamer_pipeline(video_path, model_size, confidence)
        if success:
            log.info("DL Streamer pipeline completed successfully.")
            log.info("Pipeline output:\n%s", output)
            # DL Streamer writes its own FPS stats; no matplotlib panel needed
            # for the GStreamer path (frames never touch Python).
            return
        log.warning("DL Streamer pipeline failed — falling back to OpenCV.\n%s", output)

    _run_opencv_fallback(video_path, model_size, confidence)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="License Plate Detection & OCR (3-phase pipeline).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image (used by Phase 1 and Phase 2).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to an input video (used by Phase 3).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only a specific phase (default: run all applicable).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=VALID_SIZES,
        default=DEFAULT_SIZE,
        help="YOLO model size (default: %(default)s).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Detection confidence threshold (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.image is None and args.video is None:
        parser.error("Provide at least --image (Phase 1/2) or --video (Phase 3).")

    phases_to_run: list[int] = []
    if args.phase is not None:
        phases_to_run = [args.phase]
    else:
        if args.image:
            phases_to_run.extend([1, 2])
        if args.video:
            phases_to_run.append(3)

    for phase in phases_to_run:
        if phase in (1, 2) and args.image is None:
            log.error("Phase %d requires --image.", phase)
            continue
        if phase == 3 and args.video is None:
            log.error("Phase 3 requires --video.")
            continue

        if phase == 1:
            run_phase1(args.image, args.model_size, args.confidence)
        elif phase == 2:
            run_phase2(args.image, args.model_size, args.confidence)
        elif phase == 3:
            run_phase3(args.video, args.model_size, args.confidence)

    log.info("All requested phases complete.")


if __name__ == "__main__":
    main()
