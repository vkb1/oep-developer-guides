# License Plate Detection & OCR — Sample Application

A three-phase license plate detection and OCR pipeline that progressively
moves from native PyTorch inference to Intel-optimized OpenVINO and
DL Streamer acceleration.

| Phase | Input | Detection Model | OCR Engine | Runtime |
|-------|-------|-----------------|------------|---------|
| 1 | Image | YOLO `.pt` (native PyTorch) | PaddleOCR v3.x | CPU / PyTorch |
| 2 | Image | YOLO OpenVINO IR (directory) | PaddleOCR v3.x | OpenVINO CPU/GPU |
| 3 | Video | YOLO OpenVINO IR (`.xml`) | gvaclassify / PaddleOCR fallback | DL Streamer / OpenCV |

---

## Prerequisites

- Python 3.10+
- (Phase 3, optional) Intel DL Streamer with GStreamer

---

## 1. Install Dependencies

```bash
cd sample-apps/license-plate-detection/
pip install -r requirements.txt
```

---

## 2. Download Models & Convert YOLO to OpenVINO IR

The `download_and_convert_openvino.py` script handles two tasks:

1. **Downloads** all five YOLO v11 license-plate `.pt` checkpoints
   (`v1n`, `v1s`, `v1m`, `v1l`, `v1x`) from
   [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
2. **Downloads** PP-OCRv4 server recognition weights (`inference.onnx`,
   `inference.pdiparams`) from
   [PaddlePaddle/PP-OCRv4_server_rec](https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec)
3. **Converts** each YOLO `.pt` → OpenVINO IR via
   `YOLO().export(format="openvino", imgsz=640)` and **verifies** with
   `openvino.Core().read_model()`

```bash
python download_and_convert_openvino.py
```

### CLI options

| Flag | Description |
|------|-------------|
| `--models-dir PATH` | Override the default `models/` directory |
| `--skip-download` | Skip downloading; only run the conversion step |
| `--skip-convert` | Skip YOLO → OpenVINO conversion; only download |

After this step the `models/` tree looks like:

```
models/
  morsetechlab_yolov11-license-plate-detection/
    license-plate-finetune-v1n.pt
    license-plate-finetune-v1s.pt
    license-plate-finetune-v1m.pt
    license-plate-finetune-v1l.pt
    license-plate-finetune-v1x.pt
    models_ov/
      license-plate-finetune-v1n_openvino_model/
      license-plate-finetune-v1s_openvino_model/
      license-plate-finetune-v1m_openvino_model/   ← default model
      license-plate-finetune-v1l_openvino_model/
      license-plate-finetune-v1x_openvino_model/
  PaddlePaddle_PP-OCRv4_server_rec/
    inference.onnx
    inference.pdiparams
```

---

## 3. (Optional) Convert PP-OCRv4 to OpenVINO IR

> **This is a manual step.** It is only required for the Phase 3
> DL Streamer `gvaclassify` element. Phases 1 and 2 use PaddleOCR
> directly and do **not** need this conversion.

```bash
cd models/PaddlePaddle_PP-OCRv4_server_rec/

# 1. Install paddlex and the paddle2onnx plugin
pip install paddlex==3.4.3
paddlex --install paddle2onnx

# 2. Convert Paddle → ONNX (opset 7 required for OVC compatibility)
paddlex \
    --paddle2onnx \
    --paddle_model_dir . \
    --onnx_model_dir   . \
    --opset_version    7

# 3. Convert ONNX → OpenVINO IR
#    Input shape: batch=1, channels=3, height=48, width=320
ovc ./inference.onnx \
    --output_model pp_ocrv4_server_rec.xml \
    --input "x[1,3,48,320]"
```

Expected outputs: `pp_ocrv4_server_rec.xml` and `pp_ocrv4_server_rec.bin`
inside `models/PaddlePaddle_PP-OCRv4_server_rec/`.

> **Note:** The fixed input shape `[1,3,48,320]` matches the PP-OCRv4
> server recognition model's expected crop size. Do not change it.

---

## 4. Run the Application

### Phase 1 — Native `.pt` + PaddleOCR (image)

```bash
python license_plate_app.py --image car.jpg --phase 1
```

Uses the YOLO `.pt` checkpoint directly with PaddleOCR for text
recognition. Produces a 3-panel visualisation in `output/phase1/`.

### Phase 2 — OpenVINO IR + PaddleOCR (image)

```bash
python license_plate_app.py --image car.jpg --phase 2
```

Uses the OpenVINO IR **directory** for YOLO detection and PaddleOCR for
OCR. Produces output in `output/phase2/`.

### Phase 3 — DL Streamer / OpenCV fallback (video)

```bash
python license_plate_app.py --video traffic.mp4 --phase 3
```

If Intel DL Streamer is installed, a full GStreamer pipeline
(`gvadetect` + `gvaclassify`) is used. Otherwise the application
automatically falls back to OpenCV frame-by-frame processing with YOLO
OpenVINO IR detection and PaddleOCR (OCR runs every 10 detection
frames). Produces output in `output/phase3/`.

### Run all applicable phases at once

```bash
python license_plate_app.py --image car.jpg --video traffic.mp4
```

This runs Phase 1 and 2 on the image, and Phase 3 on the video.

### Additional options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size {v1n,v1s,v1m,v1l,v1x}` | YOLO model size | `v1m` |
| `--confidence FLOAT` | Detection confidence threshold | `0.25` |

---

## Output

Each phase saves a **3-panel side-by-side figure** to its output
directory:

1. **Original** input image/frame
2. **Annotated** image with bounding boxes and confidence scores
3. **Metrics** panel with detection time, OCR time, total time, plates
   detected, recognised text, and bounding box coordinates

```
output/
  phase1/result_car.png
  phase2/result_car.png
  phase3/result_traffic.png
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ConvertPirAttribute2RuntimeAttribute not support` | PaddlePaddle OneDNN | Set `enable_mkldnn=False` (already configured) |
| `is not a supported model format` (Ultralytics) | Passing `.xml` to `YOLO()` | Pass the *directory* instead |
| `Could not open saved_model.pb` (gvadetect) | Passing directory to gvadetect | Pass the `.xml` file directly |
| `Creating model: (PP-LCNet...)` repeating per frame | PaddleOCR re-instantiated in loop | Create one instance before the loop |
| OCR/detection time inflated vs Phase 1 | No warmup run | Add warmup before `time.perf_counter()` |
| PP-OCRv5 models downloaded instead of local PP-OCRv4 | `rec_model_dir` not set | Set `rec_model_dir=str(PPOCR_MODEL_DIR)` |
| `gst-launch-1.0: command not found` | DL Streamer env not sourced | Source `setup_dls_env.sh`; the app does this automatically |
