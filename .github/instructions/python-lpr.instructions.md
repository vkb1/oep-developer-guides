---
description: "Use when writing or editing Python LPR scripts (lpr_native.py, lpr_openvino.py, download_models.py, utils.py). Covers model loading, inference patterns, and display conventions."
applyTo: "**/*.py"
---

# Python LPR Coding Guidelines

## Model Paths

- Use `BASE_DIR = Path(__file__).resolve().parent` as root for all paths
- YOLO native: `models/yolov11-license-plate/license-plate-finetune-v1m.pt`
- YOLO OpenVINO: `models_ov/license-plate-finetune-v1m_openvino_model/` (directory, not `.xml`)
- OCR native: `models/PP-OCRv4_server_rec/` (PaddlePaddle 3.x format: `.json` + `.pdiparams`)
- OCR OpenVINO: `models_ov/PP-OCRv4_server_rec/PP-OCRv4_server_rec.xml`
- Character dict: `models_ov/PP-OCRv4_server_rec/ppocr_keys_v1.txt`

## Ultralytics OpenVINO Loading

```python
from ultralytics import YOLO
# Directory name MUST end with _openvino_model
model = YOLO(str(YOLO_OV_DIR), task="detect")
```

Do NOT pass `.xml` path directly — Ultralytics requires the directory path.

## OCR OpenVINO Loading

```python
core = ov.Core()
core.set_property({"CACHE_DIR": str(BASE_DIR / "models_ov" / ".cache")})
compiled_model = core.compile_model(str(OCR_XML), "CPU")
```

Use `get_partial_shape()` for dynamic shapes — never call `.shape` directly on dynamic inputs.

## Display on Headless Servers

```python
import matplotlib
matplotlib.use("Agg")  # Must be before any pyplot import

# For display, always wrap cv2.imshow:
try:
    cv2.imshow("Title", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error:
    print("(No display available — view the saved image directly.)")
```

## Performance

- Export YOLO with `dynamic=False, imgsz=640` — never `dynamic=True`
- Add warmup inference before timed runs in OpenVINO scripts
- Enable model caching via `core.set_property({"CACHE_DIR": ...})`
