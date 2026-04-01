"""Configuration constants for the Smart Parking application."""

import os
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"

# --- HuggingFace Model Repositories ---
YOLO_LP_DETECTION_REPO = "morsetechlab/yolov11-license-plate-detection"
YOLO_LP_DETECTION_FILENAME = "yolov11-license-plate-detection.pt"

PPOCR_REC_REPO = "PaddlePaddle/PP-OCRv4_server_rec"

# --- Default Model Paths (after download) ---
YOLO_MODEL_PATH = MODELS_DIR / "yolo" / YOLO_LP_DETECTION_FILENAME
PPOCR_MODEL_DIR = MODELS_DIR / "ppocr"

# --- OpenVINO IR Model Paths ---
OV_YOLO_MODEL_DIR = MODELS_DIR / "ov_yolo"
OV_YOLO_MODEL_PATH = OV_YOLO_MODEL_DIR / "yolov11_lp_detection.xml"
OV_PPOCR_MODEL_DIR = MODELS_DIR / "ov_ppocr"

# --- DL Streamer Paths ---
DLSTREAMER_OUTPUT_DIR = OUTPUT_DIR / "dlstreamer"

# --- Inference Defaults ---
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_DEVICE = "CPU"

# --- Visualization ---
FIGURE_DPI = 100
FIGURE_SIZE = (18, 6)

# --- DL Streamer Pipeline Defaults ---
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_FRAMERATE = 30


def ensure_directories():
    """Create required directories if they don't exist."""
    for directory in [MODELS_DIR, OUTPUT_DIR, DATA_DIR,
                      YOLO_MODEL_PATH.parent, PPOCR_MODEL_DIR,
                      OV_YOLO_MODEL_DIR, OV_PPOCR_MODEL_DIR,
                      DLSTREAMER_OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
