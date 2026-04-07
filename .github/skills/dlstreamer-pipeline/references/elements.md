# DL Streamer Element Reference

## Inference Elements

| Element | Purpose | Notes |
|---------|---------|-------|
| `gvadetect` | Object detection | Works with YOLO OpenVINO models |
| `gvaclassify` | Classification on detected regions | **NOT for seq-to-seq OCR models** |
| `gvainference` | Generic inference | For custom models |

## Annotation Elements

| Element | Purpose |
|---------|---------|
| `gvawatermark` | Draw detection boxes/labels on frames |
| `gvafpscounter` | Print FPS to terminal |
| `gvametaconvert` | Convert metadata to JSON |
| `gvametapublish` | Publish metadata to file/MQTT/Kafka |

## Source Elements

| Source Type | GStreamer Element |
|-------------|------------------|
| File | `filesrc location=path.mp4 ! decodebin3` |
| URL/RTSP | `urisourcebin buffer-size=4096 uri=URL` |
| Camera | `v4l2src device=/dev/video0` |

## Sink Elements

| Output | Pipeline Suffix |
|--------|----------------|
| AVI file (uncompressed) | `avimux ! filesink location=out.avi` |
| MP4 file (compressed) | `x264enc ! h264parse ! mp4mux ! filesink location=out.mp4` |
| Display window | `autovideosink sync=false` |

## Device Configuration

| Device | Decode | Pre-process |
|--------|--------|-------------|
| CPU | `decodebin3` | `pre-process-backend=opencv` |
| GPU | `decodebin3 ! vapostproc ! video/x-raw(memory:VAMemory)` | `pre-process-backend=va-surface-sharing` |
