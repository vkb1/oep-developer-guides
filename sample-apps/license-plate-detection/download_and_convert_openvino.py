#!/usr/bin/env python3
"""Download YOLO license-plate models and PP-OCRv4 weights, then convert
YOLO .pt files to OpenVINO IR format.

Usage
-----
    python download_and_convert_openvino.py          # default models dir
    python download_and_convert_openvino.py --models-dir /path/to/models

The script downloads:
  * All five YOLO v11 license-plate .pt checkpoints (v1n, v1s, v1m, v1l, v1x)
    from  morsetechlab/yolov11-license-plate-detection
  * PP-OCRv4 server recognition weights (inference.onnx, inference.pdiparams)
    from  PaddlePaddle/PP-OCRv4_server_rec

Each .pt file is then exported to OpenVINO IR via Ultralytics' built-in
export(), and the resulting IR is verified with openvino.Core().read_model().

NOTE: PP-OCRv4 conversion to OpenVINO IR is a *manual* step that requires
paddlex + paddle2onnx + ovc.  See the project README for instructions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hugging Face repository identifiers ──────────────────────────────────
YOLO_HF_REPO = "morsetechlab/yolov11-license-plate-detection"
PPOCR_HF_REPO = "PaddlePaddle/PP-OCRv4_server_rec"

# ── Model sizes available for the YOLO license-plate detector ─────────────
YOLO_SIZES = ("v1n", "v1s", "v1m", "v1l", "v1x")

# ── Default local directory layout ────────────────────────────────────────
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _download_yolo_pt_files(models_dir: Path) -> list[Path]:
    """Download every YOLO .pt checkpoint and return their local paths."""
    dest = models_dir / "morsetechlab_yolov11-license-plate-detection"
    dest.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for size in YOLO_SIZES:
        filename = f"license-plate-finetune-{size}.pt"
        log.info("Downloading %s from %s …", filename, YOLO_HF_REPO)
        local_path = hf_hub_download(
            repo_id=YOLO_HF_REPO,
            filename=filename,
            local_dir=str(dest),
        )
        pt_path = Path(local_path)
        log.info("  → saved to %s  (%d bytes)", pt_path, pt_path.stat().st_size)
        downloaded.append(pt_path)
    return downloaded


def _download_ppocr_files(models_dir: Path) -> Path:
    """Download PP-OCRv4 server recognition weights (ONNX + pdiparams)."""
    dest = models_dir / "PaddlePaddle_PP-OCRv4_server_rec"
    dest.mkdir(parents=True, exist_ok=True)

    for filename in ("inference.onnx", "inference.pdiparams"):
        log.info("Downloading %s from %s …", filename, PPOCR_HF_REPO)
        local_path = hf_hub_download(
            repo_id=PPOCR_HF_REPO,
            filename=filename,
            local_dir=str(dest),
        )
        log.info("  → saved to %s", local_path)
    return dest


def _convert_yolo_to_openvino(pt_paths: list[Path], models_dir: Path) -> None:
    """Export each .pt → OpenVINO IR and verify with openvino.Core()."""
    # Late imports so the script can parse --help without heavy deps
    from ultralytics import YOLO  # noqa: E402
    import openvino as ov  # noqa: E402

    ov_base = (
        models_dir
        / "morsetechlab_yolov11-license-plate-detection"
        / "models_ov"
    )
    ov_base.mkdir(parents=True, exist_ok=True)

    core = ov.Core()

    for pt_path in pt_paths:
        stem = pt_path.stem  # e.g. license-plate-finetune-v1m
        ov_dir = ov_base / f"{stem}_openvino_model"

        if ov_dir.exists() and (ov_dir / f"{stem}.xml").exists():
            log.info("OpenVINO IR already exists for %s – skipping export", stem)
        else:
            log.info("Exporting %s → OpenVINO IR (imgsz=640) …", pt_path.name)
            model = YOLO(str(pt_path))
            model.export(format="openvino", imgsz=640)

            # Ultralytics writes the IR next to the .pt file.  Move it into
            # the models_ov/ sub-tree so the project layout is clean.
            auto_dir = pt_path.parent / f"{stem}_openvino_model"
            if auto_dir.exists() and auto_dir != ov_dir:
                log.info("Moving %s → %s", auto_dir, ov_dir)
                if ov_dir.exists():
                    import shutil
                    shutil.rmtree(ov_dir)
                auto_dir.rename(ov_dir)

        # ── Verify the IR ────────────────────────────────────────────────
        xml_path = ov_dir / f"{stem}.xml"
        if xml_path.exists():
            log.info("Verifying %s …", xml_path)
            ir_model = core.read_model(str(xml_path))
            log.info(
                "  ✓ %s  inputs=%s  outputs=%s",
                xml_path.name,
                [inp.get_any_name() for inp in ir_model.inputs],
                [out.get_any_name() for out in ir_model.outputs],
            )
        else:
            log.warning("Expected %s not found – verify manually.", xml_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YOLO + PP-OCRv4 models and convert YOLO to OpenVINO IR."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory for downloaded models (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and only run the conversion step.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the YOLO → OpenVINO conversion step.",
    )
    args = parser.parse_args()

    models_dir: Path = args.models_dir.resolve()
    log.info("Models directory: %s", models_dir)

    # ── 1. Download ──────────────────────────────────────────────────────
    if not args.skip_download:
        pt_paths = _download_yolo_pt_files(models_dir)
        _download_ppocr_files(models_dir)
    else:
        log.info("Skipping download (--skip-download).")
        yolo_dir = models_dir / "morsetechlab_yolov11-license-plate-detection"
        pt_paths = sorted(yolo_dir.glob("license-plate-finetune-v1*.pt"))
        if not pt_paths:
            log.error("No .pt files found in %s – cannot convert.", yolo_dir)
            sys.exit(1)

    # ── 2. Convert YOLO → OpenVINO ───────────────────────────────────────
    if not args.skip_convert:
        _convert_yolo_to_openvino(pt_paths, models_dir)
    else:
        log.info("Skipping conversion (--skip-convert).")

    log.info("Done.")


if __name__ == "__main__":
    main()
