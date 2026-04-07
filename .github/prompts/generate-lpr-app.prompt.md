---
description: "Generate the complete License Plate Recognition multi-stage application from scratch, incorporating all known fixes and patterns."
agent: "lpr-build"
tools: [read, edit, search, execute, web]
---

Create the complete LPR application following [the plan](./plan-licensePlateRecognition.prompt.md).

## Requirements

Generate these files incorporating ALL known fixes from development history:

1. **requirements.txt** — Include `paddle2onnx`, `onnxruntime>=1.10.0`, `paddleocr>=2.9`
2. **download_models.py** — YOLO export with `dynamic=False, imgsz=640`; OCR via paddle2onnx → ovc with `--input "x[1,3,48,320]"`
3. **utils.py** — Timer, draw_detections, crop_plate_region, create_side_by_side_figure, format_inference_stats
4. **lpr_native.py** — YOLO .pt + PaddleOCR API; `matplotlib.use("Agg")`; cv2.imshow in try/except
5. **lpr_openvino.py** — Ultralytics OV (dir ending `_openvino_model`); warmup inference; `get_partial_shape()` for OCR; model caching
6. **lpr_dlstreamer.py** — Detection-only `gvadetect` pipeline (NO `gvaclassify`); `source setup_dls_env.sh &&` in subprocess; `avimux` output
7. **.gitignore** — Include `models/`, `models_ov/`, `output/`, `venv/`, `__pycache__/`

Validate all files compile with `python -m py_compile`.
