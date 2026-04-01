# V1 Sample App: License Plate Detection using HuggingFace Models

## Overview

This application uses HuggingFace models to detect license plates in car images and
extract text from the detected plates. It provides a side-by-side visual comparison
showing the original image, detection results, and extracted text with inference
timing statistics.

## Models Used

| Model | Source | Purpose |
|-------|--------|---------|
| YOLOv11 | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | License plate detection |
| TrOCR Base | [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) | Printed text recognition |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models (Optional)

Models are automatically downloaded from HuggingFace on first run. To pre-download:

```bash
python download_models.py --output-dir ./models
```

## Usage

### Basic Usage

```bash
python v1_license_plate_detection.py <path_to_car_image>
```

### Advanced Options

```bash
python v1_license_plate_detection.py <image_path> \
    --yolo-model morsetechlab/yolov11-license-plate-detection \
    --trocr-model microsoft/trocr-base-printed \
    --confidence 0.25 \
    --output-dir ./output
```

### Using Pre-downloaded Models

```bash
python v1_license_plate_detection.py <image_path> \
    --yolo-model ./models/yolo \
    --trocr-model ./models/trocr
```

## Output

The application generates:

1. **Annotated Image** (`output/v1_annotated.jpg`): Input image with bounding boxes
   and recognized text overlaid on detected license plates.

2. **Comparison Figure** (`output/v1_comparison.png`): Three-panel figure showing:
   - Original input image
   - License plate detection results with bounding boxes
   - Extracted text and performance metrics

## Pipeline Architecture

```
Input Image → YOLOv11 Detection → Crop License Plates → TrOCR OCR → Results
                  │                                          │
                  ├── Detection Time                         ├── OCR Time per Plate
                  └── Bounding Boxes                         └── Extracted Text
```

## Performance Metrics

The application reports:
- **Detection inference time**: Time for YOLO model to detect license plates
- **OCR inference time**: Time for TrOCR to extract text from each plate
- **Total pipeline time**: End-to-end processing time
- **Detection confidence**: Confidence score for each detected plate
