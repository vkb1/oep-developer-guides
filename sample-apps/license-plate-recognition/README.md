# License Plate Recognition Sample Applications

A collection of progressive sample applications demonstrating license plate
detection and text recognition using different optimization approaches, from
HuggingFace models to OpenVINO optimization to Intel DL Streamer pipelines.

## Sample Applications

| Version | Description | Models | Key Features |
|---------|-------------|--------|-------------|
| [V1](v1_sample_app/) | HuggingFace Models | YOLOv11 + TrOCR | Baseline detection & OCR with inference timing |
| [V2](v2_sample_app/) | OpenVINO Optimized | YOLOv11-OV + TrOCR-OV | Model conversion to IR format, performance comparison |
| [V3](v3_sample_app/) | DL Streamer Pipeline | YOLOv11-OV + TrOCR-OV | GStreamer pipeline for real-time video processing |

## Quick Start

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install per-version dependencies
pip install -r v1_sample_app/requirements.txt  # V1 only
pip install -r v2_sample_app/requirements.txt  # V2 only
pip install -r v3_sample_app/requirements.txt  # V3 only
```

### Run V1: HuggingFace Models

```bash
cd v1_sample_app

# Download models (optional - auto-downloaded on first run)
python download_models.py

# Run detection on a car image
python v1_license_plate_detection.py path/to/car_image.jpg
```

### Run V2: OpenVINO Optimized

```bash
cd v2_sample_app

# Convert models to OpenVINO IR format
python convert_models.py

# Run detection with optimized models
python v2_license_plate_detection.py path/to/car_image.jpg

# Run with V1 comparison
python v2_license_plate_detection.py path/to/car_image.jpg --compare-v1
```

### Run V3: DL Streamer Pipeline

```bash
cd v3_sample_app

# Using shell script (requires DL Streamer)
chmod +x license_plate_recognition.sh
./license_plate_recognition.sh

# Using Python (falls back to OpenCV if DL Streamer is not available)
python v3_dlstreamer_pipeline.py --input path/to/video.mp4
```

## Project Structure

```
license-plate-recognition/
├── README.md                          # This file
├── requirements.txt                   # Common dependencies
├── sample_images/                     # Sample test images
│   └── .gitkeep
├── v1_sample_app/                     # V1: HuggingFace models
│   ├── README.md
│   ├── requirements.txt
│   ├── download_models.py             # Model download utility
│   └── v1_license_plate_detection.py  # Main V1 application
├── v2_sample_app/                     # V2: OpenVINO optimized
│   ├── README.md
│   ├── requirements.txt
│   ├── convert_models.py              # Model conversion utility
│   └── v2_license_plate_detection.py  # Main V2 application
└── v3_sample_app/                     # V3: DL Streamer pipeline
    ├── README.md
    ├── requirements.txt
    ├── license_plate_recognition.sh   # DL Streamer shell script
    └── v3_dlstreamer_pipeline.py      # Python pipeline application
```

## Models

### V1 & V2 Models (HuggingFace)

| Model | Source | Task |
|-------|--------|------|
| YOLOv11 | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | License plate detection |
| TrOCR Base | [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) | Printed text recognition |

### V3 Models (same as V2, OpenVINO IR format)

| Model | Source | Task |
|-------|--------|------|
| YOLOv11 (OpenVINO IR) | Converted from [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | License plate detection |
| TrOCR Base (OpenVINO IR) | Converted from [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) | Printed text recognition |

## Pipeline Architecture

### V1 & V2: Image Processing Pipeline

```
Input Image → YOLO Detection → Crop Plates → TrOCR OCR → Results
```

### V3: Video Stream Pipeline (DL Streamer)

```
Video Stream → Decode → gvadetect (Detection) → gvaclassify (OCR) → Output
                              │                        │
                              └── License Plates ──────┘── Annotations + FPS
```
