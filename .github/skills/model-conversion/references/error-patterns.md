# Model Conversion Error Patterns

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `Could not find exportable PaddlePaddle model` | PP-OCRv4 uses PaddlePaddle 3.x format (`.json` + `.pdiparams`), not `.pdmodel` | Use `paddle2onnx.export()` → ONNX, then `ovc` → OpenVINO |
| `paddle2onnx.export() TypeError` | Passing bytes instead of file path strings | Use `model_filename=str(path)`, `params_filename=str(path)` as named arguments |
| `not a supported model format` (Ultralytics) | OpenVINO directory name doesn't end with `_openvino_model` | Rename directory: `mv old_name name_openvino_model` |
| `to_shape was called on a dynamic shape` | Calling `.shape` on dynamic OpenVINO input | Use `get_partial_shape()` with `.is_static` checks; default to h=48, w=320 |
| `ovc: command not found` | OpenVINO dev tools not installed | `pip install openvino-dev` or ensure `openvino>=2026.0.0` is installed |
| OpenVINO 25x slower than native YOLO | YOLO exported with `dynamic=True` | Re-export with `dynamic=False, imgsz=640`; delete old model dir first |
