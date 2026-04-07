---
description: "Use when building or debugging DL Streamer GStreamer pipelines, troubleshooting gvadetect/gvaclassify errors, or configuring video output."
---

# DL Streamer Pipeline Guidelines

## Environment Setup

DL Streamer must be sourced before any pipeline runs:
```bash
source /opt/intel/dlstreamer/scripts/setup_dls_env.sh
```

For subprocess execution from Python:
```python
full_cmd = f"source /opt/intel/dlstreamer/scripts/setup_dls_env.sh && {pipeline_str}"
subprocess.run(full_cmd, shell=True, executable="/bin/bash", check=True)
```

Use `executable="/bin/bash"` — `source` is a bash builtin, not available in plain `sh`.

## PP-OCRv4 Incompatibility with gvaclassify

**PP-OCRv4_server_rec is NOT compatible with `gvaclassify`.**

- `gvaclassify` expects a classification model (single-label output)
- PP-OCRv4 is a text recognition model (sequence-to-sequence / CTC)
- Attempting to use it produces: `OpenCV error: inv_scale_x > 0 in function 'resize'`

**Solution**: Use detection-only pipeline with `gvadetect`; do OCR post-processing in Python.

## Working Detection-Only Pipeline

```bash
gst-launch-1.0 \
  filesrc location=video.mp4 ! decodebin3 ! videoconvert ! \
  gvadetect model=yolo.xml device=CPU pre-process-backend=opencv ! queue ! \
  gvawatermark ! videoconvert ! avimux ! filesink location=output.avi
```

## Encoder Availability

- `x264enc` requires `gstreamer1.0-plugins-ugly` — may not be installed
- Use `avimux ! filesink` for uncompressed AVI output (always available)
- For compressed output, install: `sudo apt-get install gstreamer1.0-plugins-ugly`

## GPU Fallback

```python
if device == "GPU" and os.path.exists("/dev/dri/renderD128"):
    decode = "decodebin3 ! vapostproc ! video/x-raw\\(memory:VAMemory\\)"
    preproc = "pre-process-backend=va-surface-sharing"
else:
    decode = "decodebin3"
    preproc = "pre-process-backend=opencv"
```
