# V3 Sample App: License Plate Recognition using Intel DL Streamer

## Overview

This application adapts the Intel DL Streamer license plate recognition pipeline
for processing smart parking video streams. It uses the same OpenVINO IR optimized
models from V2 (YOLOv11 for detection and TrOCR for OCR) and displays the input
and annotated output streams side by side with FPS performance numbers.

Based on: [DL Streamer License Plate Recognition Sample](https://github.com/open-edge-platform/dlstreamer/tree/main/samples/gstreamer/gst_launch/license_plate_recognition)

## Architecture

The pipeline uses a two-stage inference approach with the V2 optimized models:

```
Video Stream → Decode → License Plate Detection → OCR Classification → Output
                            (YOLOv11-OV)              (TrOCR-OV)
```

### GStreamer Pipeline Elements

| Element | Purpose |
|---------|---------|
| `gvadetect` | License plate detection using YOLOv11 (OpenVINO IR) |
| `gvaclassify` | OCR text recognition using TrOCR (OpenVINO IR) |
| `gvawatermark` | Render detection annotations on video |
| `gvafpscounter` | Measure and report FPS performance |
| `gvametaconvert` | Convert detection metadata to JSON |

## Prerequisites

### DL Streamer Installation

For the full GStreamer pipeline, install Intel DL Streamer:

```bash
# Using Docker (recommended)
docker pull intel/dlstreamer:latest

# Or install natively - see:
# https://github.com/open-edge-platform/dlstreamer
```

### Convert Models (using V2)

This app uses the same OpenVINO IR models produced by V2's conversion script.
Run the conversion from the project root:

```bash
cd v2_sample_app
pip install -r requirements.txt
python convert_models.py --output-dir ../models_ov
```

This creates optimized models in the `models_ov/` directory:
- `models_ov/yolo/` — YOLOv11 in OpenVINO IR format
- `models_ov/trocr/` — TrOCR in OpenVINO IR format

## Usage

### Shell Script (DL Streamer)

```bash
chmod +x license_plate_recognition.sh

# Default: smart parking video with display output
./license_plate_recognition.sh

# Custom video file
./license_plate_recognition.sh /path/to/video.mp4 CPU display

# Camera input
./license_plate_recognition.sh /dev/video0 CPU display

# FPS measurement only
./license_plate_recognition.sh video.mp4 CPU fps

# Save to file
./license_plate_recognition.sh video.mp4 CPU file output.mp4

# GPU acceleration
./license_plate_recognition.sh video.mp4 GPU display
```

### Python Application

The Python application provides a fallback when DL Streamer is not installed,
using OpenCV with OpenVINO backend:

```bash
pip install -r requirements.txt

# Default smart parking video
python v3_dlstreamer_pipeline.py

# Custom input
python v3_dlstreamer_pipeline.py --input /path/to/video.mp4

# GPU device
python v3_dlstreamer_pipeline.py --device GPU

# Limit duration
python v3_dlstreamer_pipeline.py --duration 60

# All options
python v3_dlstreamer_pipeline.py \
    --input video.mp4 \
    --device CPU \
    --output-mode fps \
    --duration 30 \
    --output-dir ./output
```

## Output Modes

| Mode | Description |
|------|-------------|
| `display` | Render annotated video on screen with FPS overlay |
| `fps` | Measure FPS performance only (no display) |
| `json` | Output detection metadata as JSON |
| `file` | Save annotated video to MP4 file |
| `display-and-json` | Combined display and JSON metadata output |

## Output

The Python application generates:
- **Stream Comparison** (`output/v3_comparison.png`): Side-by-side view of
  input frames and annotated output frames with FPS performance graphs.

## Pipeline Customization

### Device Selection

- **CPU**: Uses OpenCV preprocessing backend
- **GPU**: Uses VA-API surface sharing for optimal GPU memory access
- **AUTO**: Automatic device selection by OpenVINO

### Video Sources

The pipeline supports multiple input types:
- Local video files (`.mp4`, `.avi`, etc.)
- HTTP/HTTPS URLs (streaming video)
- RTSP streams
- USB cameras (`/dev/video0`, etc.)
