---
name: dlstreamer-pipeline
description: "Build and debug DL Streamer GStreamer video pipelines for inference. Use when constructing gst-launch-1.0 commands, troubleshooting gvadetect/gvaclassify errors, or configuring video output formats."
---

# DL Streamer Pipeline Skill

## When to Use
- Build GStreamer pipelines with DL Streamer inference elements
- Debug pipeline errors (gvadetect, gvaclassify, encoder issues)
- Configure video output (AVI, MP4, display)

## Prerequisites

```bash
# Install DL Streamer
sudo apt-get install intel-dlstreamer

# Source environment (required before every pipeline run)
source /opt/intel/dlstreamer/scripts/setup_dls_env.sh

# Verify
gst-inspect-1.0 gvadetect
```

## Working Detection-Only Pipeline

```bash
gst-launch-1.0 \
  filesrc location=video.mp4 ! decodebin3 ! videoconvert ! \
  gvadetect model=yolo_model.xml device=CPU pre-process-backend=opencv ! queue ! \
  gvawatermark ! videoconvert ! gvafpscounter ! \
  avimux ! filesink location=output.avi
```

## Running from Python

```python
import subprocess
DLS_ENV = "source /opt/intel/dlstreamer/scripts/setup_dls_env.sh"
full_cmd = f"{DLS_ENV} && {pipeline_str}"
subprocess.run(full_cmd, shell=True, executable="/bin/bash", check=True)
```

- Use `executable="/bin/bash"` — `source` is a bash builtin
- Use `avimux` for AVI output (no encoder dependency)
- For MP4: `x264enc ! h264parse ! mp4mux` (requires `gstreamer1.0-plugins-ugly`)

## Known Limitations

- **PP-OCRv4_server_rec is NOT compatible with `gvaclassify`**: It's a sequence-to-sequence OCR model, not a classifier. Produces `inv_scale_x > 0` OpenCV error.
- **No `x264enc`**: Install `gstreamer1.0-plugins-ugly` or use `avimux` for uncompressed AVI.
- **GPU fallback**: Check `/dev/dri/renderD128` existence; fall back to CPU if absent.

## Pipeline Element Reference

See [element reference](./references/elements.md) for available DL Streamer elements.
