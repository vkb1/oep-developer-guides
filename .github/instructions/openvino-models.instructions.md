---
description: "Use when converting models to OpenVINO IR, troubleshooting ovc/paddle2onnx conversion, or optimizing OpenVINO inference performance."
---

# OpenVINO Model Conversion & Optimization

## YOLO → OpenVINO

```python
from ultralytics import YOLO
model = YOLO("model.pt")
model.export(format="openvino", half=False, dynamic=False, imgsz=640)
```

- **CRITICAL**: Use `dynamic=False` for static shapes — `dynamic=True` causes 25x slower inference
- Output directory **must** end with `_openvino_model` for Ultralytics to recognize it
- After export, move files to `models_ov/` and clean up intermediate directories

## PaddlePaddle 3.x → OpenVINO (OCR)

PP-OCRv4_server_rec uses PaddlePaddle 3.x format (`.json` + `.pdiparams`), not the legacy `.pdmodel`.

Conversion chain: **Paddle3x → ONNX → OpenVINO**

### Step 1: Paddle3x → ONNX via paddle2onnx

```python
import paddle2onnx
paddle2onnx.export(
    model_filename=str(json_model),       # e.g. "inference.json"
    params_filename=str(pdiparams),       # e.g. "inference.pdiparams"
    save_file=str(onnx_path),
    opset_version=14,
    enable_onnx_checker=True,
)
```

- paddle2onnx v2.1.0 `export()` takes **file path strings**, not bytes
- CLI fallback: `paddle2onnx --model_dir DIR --model_filename inference.json --params_filename inference.pdiparams --save_file out.onnx`

### Step 2: ONNX → OpenVINO via ovc

```bash
ovc ./inference.onnx --output_model PP-OCRv4_server_rec.xml --input "x[1,3,48,320]"
```

- **CRITICAL**: Use static input shape `x[1,3,48,320]` — dynamic shapes cause runtime errors
- Validates both `.xml` and `.bin` are produced

## OpenVINO Dynamic Shape Handling

If a model has dynamic shapes (`[1,3,48,-1]`), use:
```python
input_shape = compiled_model.input(0).get_partial_shape()
if input_shape[2].is_static:
    target_h = input_shape[2].get_length()
```

Never call `compiled_model.input(0).shape` on dynamic shapes — it throws `ValueError`.

## Performance Checklist

1. Export YOLO with `dynamic=False, imgsz=640`
2. Add warmup inference (cold start includes kernel compilation)
3. Enable model caching: `core.set_property({"CACHE_DIR": "path"})`
4. Delete old model files and re-export after changing export settings
