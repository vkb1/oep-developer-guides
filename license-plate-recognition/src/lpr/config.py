"""Default configuration for the License Plate Recognition application."""

import os

# HuggingFace model repositories
DETECTION_MODEL_REPO = "morsetechlab/yolov11-license-plate-detection"
OCR_MODEL_REPO = "PaddlePaddle/PP-OCRv4_server_rec"

# Default directories
DEFAULT_MODELS_DIR = os.path.join("models")
DEFAULT_OUTPUT_DIR = os.path.join("output")

# Detection model settings
DETECTION_MODEL_FILENAME = "best.pt"
DETECTION_CONFIDENCE = 0.25

# OCR model settings
OCR_MODEL_PDMODEL = "inference.pdmodel"
OCR_MODEL_PDIPARAMS = "inference.pdiparams"
OCR_INPUT_HEIGHT = 48
OCR_INPUT_WIDTH = 320
OCR_DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/"
    "release/2.7/ppocr/utils/ppocr_keys_v1.txt"
)
OCR_DICT_FILENAME = "ppocr_keys_v1.txt"

# OpenVINO optimized model subdirectories
OV_DETECTION_SUBDIR = "detection_ov"
OV_OCR_SUBDIR = "ocr_ov"

# DL Streamer defaults
DEFAULT_VIDEO_URL = (
    "https://github.com/open-edge-platform/edge-ai-resources/raw/main/videos/ParkingVideo.mp4"
)
DEFAULT_DEVICE = "CPU"
