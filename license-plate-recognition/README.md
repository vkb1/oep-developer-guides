# License Plate Recognition — Multi-Stage Inference Pipeline

An integrated Python application for license plate detection and text
recognition, built with a three-stage pipeline architecture:

| Stage | Description | Models / Runtime |
|-------|-------------|-----------------|
| **1** | Native model inference | ultralytics YOLO (.pt) + OpenVINO loading PaddlePaddle (.pdmodel) on-the-fly |
| **2** | OpenVINO IR optimized inference | Both models pre-converted to OpenVINO IR (.xml/.bin) |
| **3** | DL Streamer video pipeline | GStreamer + DL Streamer elements with OpenVINO IR models |

Each stage produces a side-by-side comparison showing the input, detection
result, and recognized text (stages 1–2) or input vs. annotated video stream
with FPS numbers (stage 3).

---

## Models

| Purpose | HuggingFace Repository |
|---------|----------------------|
| License plate detection (YOLOv11) | [`morsetechlab/yolov11-license-plate-detection`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) |
| Text recognition (PP-OCRv4) | [`PaddlePaddle/PP-OCRv4_server_rec`](https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec) |

Models are downloaded automatically on first run via `huggingface_hub`.

---

## Requirements

- Python ≥ 3.10
- **Stage 3 only:** [DL Streamer](https://dlstreamer.github.io/get_started/install.html)
  and GStreamer with `gst-launch-1.0` available on `PATH`.

## Installation

```bash
cd license-plate-recognition
pip install .
```

For development (includes linting tools):

```bash
pip install -e ".[dev]"
```

---

## Usage

### Stage 1 — Native Model Inference

Runs license plate detection with ultralytics YOLO and text recognition by
loading the PaddlePaddle model directly through OpenVINO Runtime.

```bash
python -m lpr --stage 1 --input data/car.jpg
```

### Stage 2 — OpenVINO IR Optimized Inference

Converts both models to OpenVINO IR format, then runs inference using the
optimized representations. Depends on Stage 1 for model downloads.

```bash
python -m lpr --stage 2 --input data/car.jpg
```

### Stage 3 — DL Streamer Video Pipeline

Constructs and runs a GStreamer/DL Streamer pipeline for real-time video
processing. Uses the OpenVINO IR models from Stage 2. Depends on Stages 1
and 2 for model downloads and conversion.

```bash
# Process the default sample parking video
python -m lpr --stage 3

# Process a local video file
python -m lpr --stage 3 --input path/to/video.mp4 --device CPU

# Use display output (requires a display server)
python -m lpr --stage 3 --input path/to/video.mp4 --output-mode display
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--stage {1,2,3}` | Pipeline stage to run (required) | — |
| `--input PATH` | Input image (1-2) or video/URL (3) | Required for 1-2; parking video for 3 |
| `--models-dir DIR` | Model storage directory | `models` |
| `--output-dir DIR` | Output directory | `output` |
| `--device {CPU,GPU,AUTO}` | Inference device | `CPU` |
| `--output-mode MODE` | Stage 3 mode: `display`, `fps`, `json`, `display-and-json`, `file` | `json` |
| `--verbose`, `-v` | Enable debug logging | off |

---

## Output

Each stage writes a comparison image to the output directory:

| Stage | Output File |
|-------|-------------|
| 1 | `output/stage1_comparison.png` |
| 2 | `output/stage2_comparison.png` |
| 3 | `output/stage3_comparison.png` |

**Stages 1 & 2** produce a three-panel figure:
- Input image
- Detection result with bounding boxes
- Recognized license plate text with inference timing

**Stage 3** produces a two-panel figure:
- Input video frame
- Annotated output frame with FPS numbers

---

## Project Structure

```
license-plate-recognition/
├── pyproject.toml              # Build configuration and dependencies
├── README.md                   # This file
├── src/
│   └── lpr/
│       ├── __init__.py         # Package metadata
│       ├── __main__.py         # CLI entry point
│       ├── config.py           # Default settings and constants
│       ├── downloader.py       # HuggingFace model download utilities
│       ├── stage1.py           # Stage 1: native inference
│       ├── stage2.py           # Stage 2: OpenVINO IR inference
│       ├── stage3.py           # Stage 3: DL Streamer pipeline
│       └── visualization.py    # Side-by-side comparison rendering
├── data/                       # Sample input images / videos
└── output/                     # Generated comparison outputs
```

## DL Streamer Pipeline (Stage 3)

Stage 3 adapts the
[license_plate_recognition](https://github.com/open-edge-platform/dlstreamer/tree/main/samples/gstreamer/gst_launch/license_plate_recognition)
sample from the DL Streamer repository. The GStreamer pipeline is:

```
filesrc/urisourcebin → decodebin3 → gvadetect (YOLO) → gvaclassify (OCR)
→ gvametaconvert → gvametapublish / gvafpscounter / autovideosink
```

The default input video is the
[ParkingVideo.mp4](https://github.com/open-edge-platform/edge-ai-resources/raw/main/videos/ParkingVideo.mp4)
sample from edge-ai-resources.

---

## License

Apache-2.0 — see [LICENSE](../LICENSE).
