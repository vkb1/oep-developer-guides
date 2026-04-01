# V2 Sample App: License Plate Detection using OpenVINO Optimized Models

## Overview

This application converts the HuggingFace models from V1 to OpenVINO Intermediate
Representation (IR) format and uses the optimized models for inference. It provides
a performance comparison between the original HuggingFace models and the OpenVINO
optimized versions.

## Models Used

| Model | Original Source | Optimization |
|-------|----------------|-------------|
| YOLOv11 | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | Exported via `ultralytics` YOLO export to OpenVINO format |
| TrOCR Base | [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) | Converted via `optimum-intel` or ONNX → OpenVINO IR |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Convert Models to OpenVINO IR

```bash
python convert_models.py
```

This creates optimized models in the `models_ov/` directory:
- `models_ov/yolo/` — YOLOv11 in OpenVINO IR format
- `models_ov/trocr/` — TrOCR in OpenVINO IR format

#### Conversion Options

```bash
# Convert only YOLO model
python convert_models.py --yolo-only

# Convert only TrOCR model
python convert_models.py --trocr-only

# Custom output directory
python convert_models.py --output-dir ./custom_models
```

## Usage

### Basic Usage

```bash
python v2_license_plate_detection.py <path_to_car_image>
```

### With V1 Comparison

```bash
python v2_license_plate_detection.py <image_path> --compare-v1
```

### Advanced Options

```bash
python v2_license_plate_detection.py <image_path> \
    --yolo-ov-model ./models_ov/yolo \
    --trocr-ov-model ./models_ov/trocr \
    --confidence 0.25 \
    --output-dir ./output \
    --compare-v1
```

## Output

The application generates:

1. **Annotated Image** (`output/v2_annotated.jpg`): Image with detection overlays.

2. **Comparison Figure** (`output/v2_comparison.png`): Three-panel figure showing:
   - Original input image
   - OpenVINO optimized detection results
   - Performance metrics and V1 vs V2 comparison (if `--compare-v1` is used)

## Optimization Details

### OpenVINO IR Conversion

The conversion process optimizes models for Intel hardware:

1. **YOLO Export**: Uses `ultralytics` built-in export to OpenVINO format, which
   handles model graph optimization and weight conversion.

2. **TrOCR Conversion**: Uses `optimum-intel` for direct HuggingFace-to-OpenVINO
   conversion, preserving the encoder-decoder architecture. Falls back to manual
   ONNX export with OpenVINO conversion if `optimum-intel` is not available.

### Expected Performance Gains

- **YOLO Detection**: 1.5-3x speedup on Intel CPUs with OpenVINO
- **TrOCR OCR**: 1.5-2.5x speedup with OpenVINO optimizations
- Additional gains possible with INT8 quantization (not included in this sample)
