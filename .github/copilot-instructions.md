# Project Guidelines

## Overview

License Plate Recognition (LPR) multi-stage application using YOLOv11-m for plate detection and PP-OCRv4 for text recognition, with three inference backends: native, OpenVINO, and DL Streamer.

## Architecture

```
download_models.py  → downloads HF models, converts to OpenVINO IR
lpr_native.py       → Stage 1: native YOLO .pt + PaddleOCR API
lpr_openvino.py     → Stage 2: OpenVINO runtime + CTC decode
lpr_dlstreamer.py   → Stage 3: gst-launch-1.0 detection pipeline
utils.py            → shared: Timer, visualization, drawing helpers
```

## Build and Test

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python download_models.py          # Phase 0: download + convert models
python lpr_native.py --image car.jpeg
python lpr_openvino.py --image car.jpeg
source /opt/intel/dlstreamer/scripts/setup_dls_env.sh
python lpr_dlstreamer.py --video ParkingVideo.mp4 --device CPU
```

## Conventions

- All model paths use `pathlib.Path` relative to `BASE_DIR = Path(__file__).resolve().parent`
- YOLO OpenVINO directory **must** end with `_openvino_model` suffix for Ultralytics detection
- OCR OpenVINO model **must** be exported with static input shape `x[1,3,48,320]` via `ovc`
- Matplotlib backend set to `Agg` (headless); display via `cv2.imshow` wrapped in `try/except cv2.error`
- DL Streamer env sourced inside subprocess: `source /opt/intel/dlstreamer/scripts/setup_dls_env.sh && <pipeline>`

## Known Constraints

- PP-OCRv4_server_rec is a sequence-to-sequence OCR model — incompatible with DL Streamer `gvaclassify`
- PaddlePaddle 3.x format uses `.json` + `.pdiparams` (not `.pdmodel`) — requires `paddle2onnx` → ONNX → `ovc` chain
- YOLO must be exported with `dynamic=False, imgsz=640` for OpenVINO performance; add warmup inference before timed run
- No X11 display on headless servers — always save output to file; wrap `cv2.imshow` in try/except
