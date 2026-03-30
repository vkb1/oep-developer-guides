# OEP Developer Guides — Smart Parking: License Plate Detection & Recognition

A generic Python application that detects license plates in images and video
streams, extracts text via OCR, and optionally optimizes inference with
[OpenVINO](https://docs.openvino.ai/) and
[DL Streamer](https://dlstreamer.github.io/).

## Features

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| **1** | Native HuggingFace model inference (YOLOv11 + PaddleOCR) | Image | Side-by-side comparison with inference timings |
| **2** | OpenVINO IR conversion and optimized inference | Image | Side-by-side comparison with optimized timings |
| **3** | DL Streamer / OpenCV video pipeline | Video | Input vs. annotated output with FPS metrics |

## Project Structure

```
oep-developer-guides/
├── smart_parking/
│   ├── __init__.py              # Package init
│   ├── main.py                  # CLI entry point (argparse)
│   ├── config.py                # Configuration constants and paths
│   ├── model_manager.py         # HuggingFace model download utilities
│   ├── visualization.py         # Side-by-side comparison rendering
│   ├── stage1_inference.py      # Stage 1: native YOLO + PaddleOCR
│   ├── stage2_openvino.py       # Stage 2: OpenVINO IR conversion & inference
│   └── stage3_dlstreamer.py     # Stage 3: DL Streamer / OpenCV pipeline
├── download_models.py           # Standalone script to download HuggingFace models
├── requirements.txt             # Python dependencies
├── README.md
└── LICENSE
```

## Prerequisites

- Python 3.10+
- (Optional) Intel® Distribution of OpenVINO™ toolkit for optimized inference
- (Optional) Intel® Deep Learning Streamer (DL Streamer) for GStreamer pipelines

## Installation

```bash
# Clone the repository
git clone https://github.com/vkb1/oep-developer-guides.git
cd oep-developer-guides

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

The application is run via the `smart_parking.main` module with the
`--stage` flag to select the execution stage.

### Stage 1 — Native Model Inference

Downloads models from HuggingFace (if not cached) and runs YOLO license-plate
detection + PaddleOCR text recognition on a single image.

```bash
python -m smart_parking.main --stage 1 --input data/sample_car.jpg
```

Output is saved to `output/stage1/` and includes:
- `comparison.png` — side-by-side figure (input, detection, OCR text + timings)
- `annotated.png` — image with bounding boxes
- `plate_N.png` — cropped license plate images

### Stage 2 — OpenVINO Optimized Inference

Converts the Stage 1 models to OpenVINO IR format (one-time), then runs
optimized inference on the same image.

```bash
python -m smart_parking.main --stage 2 --input data/sample_car.jpg --device CPU
```

Output is saved to `output/stage2/` with the same layout as Stage 1.

### Stage 3 — DL Streamer Video Pipeline

Processes a video file using a DL Streamer GStreamer pipeline (if available)
or an OpenCV-based fallback with per-frame detection and OCR.

```bash
python -m smart_parking.main --stage 3 --input data/parking_video.mp4 \
    --device CPU --max-frames 300
```

Output is saved to `output/stage3/` and includes:
- `annotated_output.mp4` — video with detection overlays and FPS counter
- `video_comparison.png` — side-by-side snapshot of input vs. annotated output

### Full Options

```
python -m smart_parking.main --help

options:
  --stage {1,2,3}         Execution stage
  --input INPUT           Path to input image or video
  --output-dir DIR        Custom output directory
  --confidence FLOAT      Detection confidence threshold (default: 0.5)
  --device DEVICE         OpenVINO device: CPU, GPU, etc. (default: CPU)
  --yolo-repo REPO        Custom HuggingFace repo for YOLO model
  --ppocr-repo REPO       Custom HuggingFace repo for PP-OCR model
  --yolo-model PATH       Path to local YOLO .pt model (skip download)
  --ppocr-model-dir DIR   Path to local PP-OCR model dir (skip download)
  --ov-yolo-model PATH    Path to OpenVINO YOLO .xml model (stages 2-3)
  --max-frames N          Max frames for video processing (0 = all)
  -v, --verbose           Enable debug logging
```

### Download Models

You can pre-download all required models before running any stage using the
standalone `download_models.py` script:

```bash
# Download all models (YOLO + PP-OCR)
python download_models.py

# Download only the YOLO license plate detection model
python download_models.py --yolo-only

# Download only the PP-OCR recognition model
python download_models.py --ppocr-only

# Download to a custom directory
python download_models.py --models-dir ./my_models
```

Models are downloaded from:
- **YOLO**: [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
- **PP-OCR**: [PaddlePaddle/PP-OCRv4_server_rec](https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec)

### Typical Workflow

```bash
# 1. Run Stage 1 to download models and get baseline performance
python -m smart_parking.main --stage 1 --input data/sample_car.jpg

# 2. Run Stage 2 to convert to OpenVINO and compare performance
python -m smart_parking.main --stage 2 --input data/sample_car.jpg

# 3. Run Stage 3 on a video stream
python -m smart_parking.main --stage 3 --input data/parking_video.mp4
```

## Models

| Model | Source | Purpose |
|-------|--------|---------|
| YOLOv11 License Plate Detection | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | Detects license plates in images/frames |
| PP-OCRv4 Server Recognition | [PaddlePaddle/PP-OCRv4_server_rec](https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec) | Extracts text from cropped plate regions |

Models are downloaded automatically on first run and cached in the `models/`
directory.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).