"""
Downloads and converts two models to OpenVINO IR format:

  1. morsetechlab/yolov11-license-plate-detection  (Ultralytics .pt)
     Path: YOLO .pt  ──►  OpenVINO IR   (via ultralytics export)

  2. PaddlePaddle/PP-OCRv4_server_rec  (PaddlePaddle inference model)
     Path: .pdmodel/.pdiparams  ──►  ONNX  ──►  OpenVINO IR
           (via paddle2onnx + ovc)

Directory layout produced:
  models/
  ├── morsetechlab_yolov11-license-plate-detection/
  │   ├── best.pt
  │   └── models_ov/
  │       └── best_openvino_model/
  │           ├── best.xml
  │           ├── best.bin
  │           └── metadata.yaml
  └── PaddlePaddle_PP-OCRv4_server_rec/
      ├── inference.pdmodel
      ├── inference.pdiparams
      └── models_ov/
          ├── pp_ocrv4_server_rec.onnx
          └── pp_ocrv4_server_rec_openvino_model/
              ├── pp_ocrv4_server_rec.xml
              └── pp_ocrv4_server_rec.bin

Requirements:
    pip install ultralytics huggingface_hub openvino paddle2onnx paddlepaddle
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _require(pkg_import: str, pkg_install: str) -> None:
    """Exit with an install hint if a package is missing."""
    try:
        __import__(pkg_import)
    except ImportError:
        sys.exit(f"'{pkg_import}' not found. Run: pip install {pkg_install}")


_require("huggingface_hub", "huggingface_hub")
_require("ultralytics",     "ultralytics>=8.1")
_require("openvino",        "openvino")
# _require("paddle2onnx",     "paddle2onnx")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from ultralytics import YOLO                         # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def discover_files(repo_id: str, extensions: tuple[str, ...]) -> list[str]:
    """Return all files in an HF repo matching any of the given extensions."""
    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    matched = [f for f in all_files if any(f.endswith(ext) for ext in extensions)]
    return matched


def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download one file from HF Hub into local_dir (flat, no cache nesting)."""
    print(f"  [↓] {filename}")
    raw = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        local_dir=str(local_dir),
    )
    dest = local_dir / Path(filename).name
    if Path(raw) != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, dest)
    return dest


def verify_ir(ov_model_dir: Path) -> None:
    """Load the IR with OpenVINO Core and print input/output names."""
    xml_files = list(ov_model_dir.glob("*.xml"))
    if not xml_files:
        print("    [!] No .xml found – skipping verification.")
        return
    from openvino import Core
    core = Core()
    model = core.read_model(str(xml_files[0]))
    inputs  = [i.get_any_name() for i in model.inputs]
    outputs = [o.get_any_name() for o in model.outputs]
    print(f"    [✓] IR verified  |  inputs={inputs}  outputs={outputs}")


# ---------------------------------------------------------------------------
# Model 1 – YOLOv11 license-plate detection  (.pt → OpenVINO IR)
# ---------------------------------------------------------------------------

YOLO_REPO_ID = "morsetechlab/yolov11-license-plate-detection"


def run_yolo(base_dir: Path, imgsz: int, half: bool, dynamic: bool, skip_verify: bool) -> None:
    safe_name = YOLO_REPO_ID.replace("/", "_")
    model_dir = base_dir / safe_name
    ov_dir    = model_dir / "models_ov"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[YOLO] {YOLO_REPO_ID}")
    print(f"{'='*60}")

    pt_files = discover_files(YOLO_REPO_ID, (".pt",))
    if not pt_files:
        print("  [!] No .pt files found – skipping.")
        return
    print(f"  Found {len(pt_files)} .pt file(s): {pt_files}")

    for pt_filename in pt_files:
        pt_path = download_file(YOLO_REPO_ID, pt_filename, model_dir)

        print(f"\n  [⚙] Exporting {pt_path.name} → OpenVINO IR ...")
        model = YOLO(str(pt_path))
        model.export(format="openvino", imgsz=imgsz, half=half, dynamic=dynamic)

        exported_dir = pt_path.parent / f"{pt_path.stem}_openvino_model"
        if not exported_dir.exists():
            print(f"  [!] Expected export dir not found: {exported_dir}")
            continue

        ov_dir.mkdir(parents=True, exist_ok=True)
        dest = ov_dir / exported_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(exported_dir), str(dest))
        print(f"  [✓] Saved → {dest}")

        if not skip_verify:
            verify_ir(dest)


# ---------------------------------------------------------------------------
# Model 2 – PP-OCRv4 server rec  (.pdmodel/.pdiparams → ONNX → OpenVINO IR)
# ---------------------------------------------------------------------------

PADDLE_REPO_ID   = "PaddlePaddle/PP-OCRv4_server_rec"
PADDLE_MODEL_FILE = "inference.pdmodel"
PADDLE_PARAMS_FILE = "inference.pdiparams"
PADDLE_STEM       = "pp_ocrv4_server_rec"

# Typical PP-OCRv4 rec input: [batch, channels, height, width]
# Height is fixed at 48; width can be fixed (320) or dynamic (-1).
PADDLE_INPUT_SHAPE = "x:1,3,48,320"   # change to "x:1,3,48,-1" for dynamic width


def _run(cmd: list[str], step: str) -> None:
    """Run a subprocess command, exit on failure."""
    print(f"  [⚙] {step}")
    print(f"      $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        sys.exit(f"  [✗] {step} failed (exit {result.returncode})")


def run_paddle(base_dir: Path, skip_verify: bool) -> None:
    safe_name = PADDLE_REPO_ID.replace("/", "_")
    model_dir = base_dir / safe_name
    ov_dir    = model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Paddle] {PADDLE_REPO_ID}")
    print(f"{'='*60}")

    # 1. Download all files from the PaddlePaddle repo
    paddle_files = discover_files(PADDLE_REPO_ID, ("",))  # Empty tuple to match all files
    if not paddle_files:
        print("  [!] No files found in repo – skipping.")
        return
    print(f"  Found {len(paddle_files)} file(s): {paddle_files}")

    for fname in paddle_files:
        download_file(PADDLE_REPO_ID, fname, model_dir)

    pdmodel_path  = model_dir / PADDLE_MODEL_FILE
    pdiparams_path = model_dir / PADDLE_PARAMS_FILE

    # for p in (pdmodel_path, pdiparams_path):
    #     if not p.exists():
    #         sys.exit(f"  [✗] Required file not found after download: {p}")

    ov_dir.mkdir(parents=True, exist_ok=True)

    # 2. Convert PaddlePaddle → ONNX  (paddle2onnx)
    onnx_path = ov_dir / f"{PADDLE_STEM}.onnx"
    # Parse input shape: "x:1,3,48,320" → "{'x': [1,3,48,320]}"
    shape_parts = PADDLE_INPUT_SHAPE.split(":")
    input_name = shape_parts[0]
    shape_values = shape_parts[1]
    input_shape_dict = f"{{{input_name!r}: [{shape_values}]}}"
    
    _run(
        [
            "paddle2onnx",
            "--model_dir",       str(model_dir),
            "--model_filename",  PADDLE_MODEL_FILE,
            "--params_filename", PADDLE_PARAMS_FILE,
            "--save_file",       str(onnx_path),
            "--opset_version",   "11",
            "--enable_onnx_checker", "True"
        ],
        "paddle2onnx: PaddlePaddle → ONNX",
    )

    if not onnx_path.exists():
        sys.exit(f"  [✗] ONNX file not produced: {onnx_path}")
    print(f"  [✓] ONNX saved → {onnx_path}")

    # 3. Convert ONNX → OpenVINO IR  (ovc)
    ov_model_dir = ov_dir / f"{PADDLE_STEM}_openvino_model"
    ov_model_dir.mkdir(parents=True, exist_ok=True)
    ov_xml = ov_model_dir / f"{PADDLE_STEM}.xml"

    _run(
        [
            "ovc",
            str(onnx_path),
            "--output_model", str(ov_xml),
        ],
        "ovc: ONNX → OpenVINO IR",
    )

    print(f"  [✓] OpenVINO IR saved → {ov_model_dir}")

    # 4. Verify
    if not skip_verify:
        verify_ir(ov_model_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and convert to OpenVINO IR:\n"
            "  • morsetechlab/yolov11-license-plate-detection (.pt)\n"
            "  • PaddlePaddle/PP-OCRv4_server_rec (.pdmodel)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", "-o", default="models",
        help="Root directory for all downloaded/converted models (default: ./models)",
    )
    # YOLO-specific
    yolo_grp = parser.add_argument_group("YOLOv11 export options")
    yolo_grp.add_argument("--imgsz",   type=int, default=640,  help="Input image size (default: 640)")
    yolo_grp.add_argument("--half",    action="store_true",    help="Export YOLO in FP16")
    yolo_grp.add_argument("--dynamic", action="store_true",    help="Enable dynamic batch size for YOLO")

    # Shared
    parser.add_argument("--skip-verify", action="store_false", help="Skip OpenVINO IR verification")
    parser.add_argument(
        "--model", choices=["yolo", "paddle", "all"], default="all",
        help="Which model(s) to process (default: all)",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)

    # if args.model in ("yolo", "all"):
    #     run_yolo(
    #         base_dir,
    #         imgsz=args.imgsz,
    #         half=args.half,
    #         dynamic=args.dynamic,
    #         skip_verify=args.skip_verify,
    #     )

    if args.model in ("paddle", "all"):
        run_paddle(base_dir, skip_verify=args.skip_verify)

    # Final summary
    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    print(f"Output root: {base_dir.resolve()}")
    for child in sorted(base_dir.rglob("*.xml")):
        print(f"  IR → {child}")


if __name__ == "__main__":
    main()