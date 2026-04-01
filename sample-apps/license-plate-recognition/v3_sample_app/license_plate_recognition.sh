#!/bin/bash
# =============================================================================
# V3: License Plate Recognition using Intel DL Streamer
#
# This script constructs and runs a GStreamer pipeline using DL Streamer
# elements for license plate detection and OCR recognition on video streams.
#
# Adapted from:
# https://github.com/open-edge-platform/dlstreamer/tree/main/samples/gstreamer/gst_launch/license_plate_recognition
#
# Usage:
#   ./license_plate_recognition.sh [INPUT] [DEVICE] [OUTPUT]
#
# Arguments:
#   INPUT  - Video source: file path, URL, or camera device
#            Default: Pexels smart parking video
#   DEVICE - Inference device: CPU, GPU, or AUTO
#            Default: CPU (GPU if /dev/dri/renderD128 exists)
#   OUTPUT - Output mode: display, fps, json, file, display-and-json
#            Default: display
# =============================================================================

set -euo pipefail

# ----- Configuration --------------------------------------------------------

# Use the same V2 optimized OpenVINO IR models
MODELS_PATH="${MODELS_PATH:-models_ov}"

# Models — V2 converted OV IR models (YOLOv11 + TrOCR)
# These are produced by v2_sample_app/convert_models.py
DETECTION_MODEL_DIR="${MODELS_PATH}/yolo"
OCR_MODEL_DIR="${MODELS_PATH}/trocr"

# Auto-discover .xml model files within model directories
find_model_xml() {
    local model_dir="$1"
    local xml_file
    xml_file=$(find "$model_dir" -name "*.xml" -type f 2>/dev/null | sort | head -n1)
    if [[ -z "$xml_file" ]]; then
        echo ""
    else
        echo "$xml_file"
    fi
}

DETECTION_MODEL=$(find_model_xml "$DETECTION_MODEL_DIR")
OCR_MODEL=$(find_model_xml "$OCR_MODEL_DIR")

# Default video: smart parking scene from Pexels
DEFAULT_VIDEO="https://videos.pexels.com/video-files/3014296/3014296-hd_1920_1080_24fps.mp4"

# Parse arguments
INPUT="${1:-$DEFAULT_VIDEO}"
DEVICE="${2:-CPU}"
OUTPUT="${3:-display}"

# ----- Device Configuration -------------------------------------------------

GPU_RENDER_DEVICE="/dev/dri/renderD128"

if [[ "$DEVICE" == "GPU" ]] && [[ ! -e "$GPU_RENDER_DEVICE" ]]; then
    echo "Warning: GPU render device $GPU_RENDER_DEVICE not found, falling back to CPU"
    DEVICE="CPU"
fi

if [[ "$DEVICE" == "GPU" ]]; then
    PRE_PROCESS_BACKEND="va-surface-sharing"
    DECODE="decodebin3 ! vapostproc ! video/x-raw(memory:VAMemory)"
else
    PRE_PROCESS_BACKEND="opencv"
    DECODE="decodebin3"
fi

# ----- Source Configuration -------------------------------------------------

if [[ "$INPUT" == /dev/video* ]]; then
    SOURCE="v4l2src device=$INPUT"
elif [[ "$INPUT" == http://* ]] || [[ "$INPUT" == https://* ]] || [[ "$INPUT" == rtsp://* ]]; then
    SOURCE="urisourcebin uri=\"$INPUT\""
else
    SOURCE="filesrc location=\"$INPUT\""
fi

# ----- Sink Configuration --------------------------------------------------

case "$OUTPUT" in
    display)
        SINK="gvawatermark ! gvafpscounter ! videoconvert ! \
              fpsdisplaysink video-sink=autovideosink sync=false"
        ;;
    fps)
        SINK="gvafpscounter ! fakesink sync=false"
        ;;
    json)
        SINK="gvametaconvert ! gvafpscounter ! fakesink sync=false"
        ;;
    file)
        OUTPUT_FILE="${4:-output/v3_output.mp4}"
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        SINK="gvawatermark ! gvafpscounter ! videoconvert ! \
              x264enc ! mp4mux ! filesink location=$OUTPUT_FILE"
        ;;
    display-and-json)
        SINK="tee name=t ! queue ! gvawatermark ! gvafpscounter ! \
              videoconvert ! fpsdisplaysink video-sink=autovideosink sync=false \
              t. ! queue ! gvametaconvert ! fakesink sync=false"
        ;;
    *)
        echo "Error: Unknown output mode: $OUTPUT"
        echo "Valid options: display, fps, json, file, display-and-json"
        exit 1
        ;;
esac

# ----- Validate Models ------------------------------------------------------

echo "============================================================"
echo "V3: License Plate Recognition - DL Streamer Pipeline"
echo "============================================================"
echo "Input:      $INPUT"
echo "Device:     $DEVICE"
echo "Output:     $OUTPUT"
echo "Detection:  $DETECTION_MODEL"
echo "OCR:        $OCR_MODEL"
echo "============================================================"

if [[ -z "$DETECTION_MODEL" ]] || [[ ! -f "$DETECTION_MODEL" ]]; then
    echo ""
    echo "Warning: Detection model not found in: $DETECTION_MODEL_DIR"
    echo "Please run v2_sample_app/convert_models.py first. See README.md for instructions."
    echo ""
fi

if [[ -z "$OCR_MODEL" ]] || [[ ! -f "$OCR_MODEL" ]]; then
    echo ""
    echo "Warning: OCR model not found in: $OCR_MODEL_DIR"
    echo "Please run v2_sample_app/convert_models.py first. See README.md for instructions."
    echo ""
fi

# ----- Build and Run Pipeline -----------------------------------------------

PIPELINE="$SOURCE ! \
$DECODE ! queue ! \
gvadetect model=$DETECTION_MODEL device=$DEVICE pre-process-backend=$PRE_PROCESS_BACKEND ! queue ! \
videoconvert ! \
gvaclassify model=$OCR_MODEL device=$DEVICE pre-process-backend=$PRE_PROCESS_BACKEND ! queue ! \
$SINK"

echo ""
echo "Pipeline:"
echo "gst-launch-1.0 $PIPELINE"
echo ""

# Run the pipeline
# shellcheck disable=SC2086
gst-launch-1.0 $PIPELINE
