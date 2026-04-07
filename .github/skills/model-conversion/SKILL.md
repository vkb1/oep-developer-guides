---
name: model-conversion
description: "Convert HuggingFace models to OpenVINO IR format. Use when downloading YOLO or PaddleOCR models, running paddle2onnx, executing ovc, or troubleshooting model format errors."
---

# Model Conversion Skill

## When to Use
- Download and convert YOLO/PaddleOCR models from HuggingFace
- Fix model conversion failures (PaddlePaddle 3.x format, paddle2onnx errors)
- Re-export models after changing conversion settings

## Models

| Model | Source | Native Format | OpenVINO Path |
|-------|--------|---------------|---------------|
| YOLOv11-m | `morsetechlab/yolov11-license-plate-detection` | `.pt` (40 MB) | `models_ov/license-plate-finetune-v1m_openvino_model/` |
| PP-OCRv4_server_rec | `PaddlePaddle/PP-OCRv4_server_rec` | `.json` + `.pdiparams` (90 MB) | `models_ov/PP-OCRv4_server_rec/PP-OCRv4_server_rec.xml` |

## YOLO Conversion Procedure

1. Download via `huggingface_hub.hf_hub_download(repo_id, filename)`
2. Export: `YOLO(pt_path).export(format="openvino", half=False, dynamic=False, imgsz=640)`
3. Move output to `models_ov/license-plate-finetune-v1m_openvino_model/`
4. Verify: directory name ends with `_openvino_model`

**NEVER use `dynamic=True`** — causes 25x slower OpenVINO inference.

## OCR Conversion Procedure

PaddlePaddle 3.x uses `.json` + `.pdiparams` (NOT `.pdmodel`).

1. Download via `huggingface_hub.snapshot_download(repo_id)`
2. Convert Paddle → ONNX:
   ```python
   paddle2onnx.export(
       model_filename=str(json_path),    # "inference.json"
       params_filename=str(params_path), # "inference.pdiparams"
       save_file=str(onnx_path),
       opset_version=14,
       enable_onnx_checker=True,
   )
   ```
3. Convert ONNX → OpenVINO with static shape:
   ```bash
   ovc ./inference.onnx --output_model PP-OCRv4_server_rec.xml --input "x[1,3,48,320]"
   ```
4. Download character dictionary:
   ```
   https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt
   ```
5. Clean up intermediate ONNX file

## Troubleshooting

See [error patterns reference](./references/error-patterns.md) for common conversion failures.
