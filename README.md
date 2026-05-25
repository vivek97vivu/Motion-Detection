<div align="center">

# 🎯 **RTSP Motion Detection Engine**

### 🚨 Real-Time ROI-Based Motion Detection for Smart Surveillance Systems

A **production-grade computer vision pipeline** built for **real-time IP camera / RTSP monitoring**, combining **GPU-accelerated background subtraction + intelligent alert management** for high-precision, low-noise motion events.

> ⚙️ Powered by **OpenCV MOG2 (Detection)** + **CUDA GPU Acceleration (Processing)**
> 🧠 Designed for **low false positives, high reliability deployments**
> 🧩 Engineered with a **fully modular production architecture** — config-driven, zero code changes needed

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-brightgreen?logo=nvidia&logoColor=white)](#)
[![RTSP](https://img.shields.io/badge/RTSP-TCP%20%7C%20UDP-blue)](#)
[![YAML](https://img.shields.io/badge/Config-YAML-orange)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey?logo=linux&logoColor=white)](#)

</div>

---

## ⚡ Core Stack

| Component | Purpose |
|---|---|
| 🎯 **MOG2 Background Subtractor** | Frame-by-frame foreground extraction, GPU-accelerated |
| 🎥 **NVDEC Hardware Decode** | H264/H265 RTSP decode on GPU — zero CPU decode cost |
| 🖼️ **Interactive ROI Selector** | Draw your detection zone on the live frame at startup |
| 📸 **Alert Manager** | Cooldown-gated JPEG snapshots + JSON metadata per event |
| 🔁 **Auto-Reconnect Stream** | Threaded RTSP reader — silent reconnect on camera drop |
| ⚙️ **YAML Config Engine** | All settings in `config.yaml` + `logging.yaml` — no code changes |
| 📋 **Structured Logging** | 4 rotating log files: app, error, alerts (JSON), stream |

---

## 🏗️ Project Structure

```
motion_detector/
│
├── main.py                        ← Entry point — only file you run
│
├── config/
│   ├── config.yaml                ← ALL application settings
│   └── logging.yaml               ← Log handlers, levels, rotation policy
│
├── core/
│   ├── stream.py                  ← Threaded RTSP reader + NVDEC GPU decode + auto-reconnect
│   ├── detector.py                ← MOG2 motion analysis — GPU path + CPU fallback
│   ├── alert_manager.py           ← Saves JPEG snapshots + JSON metadata with cooldown
│   └── display.py                 ← All OpenCV rendering — ROI selector, overlays, banners
│
├── utils/
│   ├── config_loader.py           ← Loads & validates config.yaml → dot-access ConfigBox
│   ├── logger.py                  ← Wires logging.yaml + JSON formatter for alerts.log
│   └── __init__.py
│
├── alerts/
│   ├── snapshots/                 ← Annotated JPEG alert images
│   └── json/                      ← JSON metadata files (one per alert)
│
├── logs/
│   ├── app.log                    ← All events, daily rotation, 30-day retention
│   ├── error.log                  ← WARNING+ only, 10MB rotating
│   ├── alerts.log                 ← JSON-line alert events (machine-parseable)
│   └── stream.log                 ← RTSP connect/reconnect events
│
└── requirements.txt
```

---

## 🔁 Detection Pipeline

```
RTSP Camera
    │
    ▼
┌─────────────────────────────────┐
│   RTSPStream (background thread) │  ← NVDEC GPU decode (or CPU fallback)
│   Always holds the latest frame  │
└────────────────┬────────────────┘
                 │ raw BGR frame
                 ▼
┌─────────────────────────────────┐
│        ROI Crop                  │  ← Only drawn zone pixels processed
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Gaussian Blur (GPU/CPU)        │  ← Removes camera noise
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   MOG2 Background Subtraction   │  ← Extracts foreground mask
│   (GPU: cv2.cuda.MOG2)          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Morphology: Open + Dilate     │  ← Removes noise, fills gaps
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Contour Filter                 │  ← Drop blobs < min_contour_area_px2
└────────────────┬────────────────┘
                 │
          ┌──────┴──────┐
          │             │
       NO motion     Motion detected
          │             │
     Next frame    ┌────┴────────────────┐
                   │                     │
            Show on screen         Save alert
            (boxes + banner)   (JPEG + JSON file)
                                 if cooldown passed
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU acceleration** (NVIDIA GPU required):

```bash
# Uninstall standard OpenCV first
pip uninstall opencv-python opencv-python-headless -y

# Install CUDA-enabled OpenCV contrib
pip install opencv-contrib-python  # then build with CUDA or use pre-built wheel
```

> See [GPU Setup](#-gpu-setup) section below for full instructions.

### 2. Configure

Edit **`config/config.yaml`** — set your camera URL:

```yaml
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1"
  transport: "tcp"   # tcp (reliable) | udp (low latency)
```

### 3. Run

```bash
python main.py
```

On launch:
1. Camera connects and fetches the first frame
2. A window opens — **drag a rectangle** over your detection zone
3. Press **Enter** to confirm — monitoring begins immediately

---

## ⚙️ Configuration Reference

### `config/config.yaml`

```yaml
camera:
  rtsp_url:              "rtsp://..."      # ← Your IP camera URL
  transport:             "tcp"             # tcp | udp
  reconnect_delay_sec:   5                 # wait before reconnect on drop
  frame_buffer_size:     1                 # always use latest frame
  connection_timeout_sec: 30               # startup timeout

detection:
  min_contour_area_px2:  500    # raise to reduce false positives (noise filter)
  blur_kernel_size:      21     # Gaussian blur strength
  dilate_iterations:     2      # fills gaps in moving objects
  mog2_history:          500    # frames for background model
  mog2_threshold:        16     # lower = more sensitive
  shadow_removal:        true   # strip shadow pixels from mask

alerts:
  snapshot_dir:   alerts/snapshots  # JPEG output folder
  json_dir:       alerts/json        # JSON metadata folder
  cooldown_sec:   5                  # min gap between two saved alerts
  jpeg_quality:   90                 # 1–100
  draw_bounding_boxes:   true
  draw_zone_on_snapshot: true
  timestamp_watermark:   true

gpu:
  enabled:       true     # use GPU for MOG2 + blur + morphology
  nvdec_decode:  true     # use NVDEC for H264/H265 RTSP decode

display:
  window_title:       "RTSP Motion Detection"
  show_status_bar:    true
  show_motion_banner: true
  zone_color_normal:  [0, 220, 0]    # BGR green — idle
  zone_color_motion:  [0, 0, 255]    # BGR red — motion
  box_color:          [0, 0, 255]    # BGR red — bounding boxes

logging:
  config_file: "config/logging.yaml"
```

---

### `config/logging.yaml` — Log Files

| File | Level | Contents | Rotation |
|---|---|---|---|
| `logs/app.log` | DEBUG+ | All application events | Daily, 30-day history |
| `logs/error.log` | WARNING+ | Errors and warnings only | 10 MB × 5 files |
| `logs/alerts.log` | INFO | **JSON-line per alert** — machine-parseable | 10 MB × 10 files |
| `logs/stream.log` | DEBUG+ | RTSP connect / reconnect events | Daily, 7-day history |

---

## 📸 Alert Output

Every confirmed motion event writes **two files** with matching timestamps:

```
alerts/
  snapshots/alert_20260525_143210_847123.jpg    ← annotated frame
  json/alert_20260525_143210_847123.json        ← structured metadata
```

### JSON Structure

```json
{
  "alert_id": 12,
  "timestamp": "2026-05-25T14:32:10.847123",
  "camera_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
  "snapshot_file": "/home/user/motion_detector/alerts/snapshots/alert_....jpg",
  "detection_zone": { "x": 120, "y": 80, "width": 400, "height": 300 },
  "motion_objects": [
    { "x": 200, "y": 150, "width": 60, "height": 90, "area_px2": 5400 }
  ],
  "object_count": 1,
  "total_motion_area_px2": 5400
}
```

### `logs/alerts.log` — JSON Lines (one per alert)

```json
{"timestamp": "2026-05-25T14:32:10Z", "level": "INFO", "logger": "motion_detector.alert", "message": "Alert saved", "alert_id": 12, "snapshot": "alerts/snapshots/alert_....jpg", "object_count": 1, "total_area": 5400}
```

---

## 🎮 Live Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit cleanly — stops stream thread, releases camera |
| `R` | Pause and redraw detection zone (resets background model) |
| `S` | Force-save a snapshot immediately (bypasses cooldown) |

---

## 🖥️ GPU Setup

### Check if your OpenCV has CUDA support

```python
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())   # 0 = no CUDA, 1+ = GPU ready
```

### Option A — Pre-built CUDA Wheel (easiest)

```bash
pip install opencv-contrib-python-headless
# Use a CUDA-enabled wheel from: https://github.com/cudawarped/opencv-python-cuda-wheels
```

### Option B — Build OpenCV from source with CUDA

```bash
cmake -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=8.6 \       # your GPU compute capability
      -D WITH_CUDNN=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      -D BUILD_opencv_python3=ON ..
make -j$(nproc)
```

### NVDEC Hardware Decode (H264/H265)

Requires FFmpeg built with `--enable-nvdec`. To verify:

```bash
ffmpeg -hwaccels | grep nvdec
```

If present, set in `config.yaml`:
```yaml
gpu:
  nvdec_decode: true
```

### GPU vs CPU — What Gets Accelerated

| Operation | CPU | GPU (CUDA) |
|---|---|---|
| H264/H265 RTSP decode | ✅ Software | ✅ NVDEC hardware unit |
| Gaussian blur | ✅ | ✅ `cv2.cuda.GaussianFilter` |
| MOG2 background subtraction | ✅ | ✅ `cv2.cuda.MOG2` |
| Threshold | ✅ | ✅ `cv2.cuda.threshold` |
| Morphological open + dilate | ✅ | ✅ `cv2.cuda.MorphologyFilter` |
| Contour detection | ✅ | ⚠️ CPU (tiny — mask already small) |

> **The GPU path downloads only the final binary mask (tiny) back to CPU — not the full frame. All heavy processing stays on GPU.**

---

## 🔧 Tuning Guide

| Symptom | Fix |
|---|---|
| Too many false alerts (leaves, lighting) | Raise `detection.min_contour_area_px2` → `1000–2000` |
| Missing slow-moving objects | Lower `detection.mog2_threshold` → `8–12` |
| Flickering on lighting changes | Raise `detection.mog2_history` → `800–1500` |
| Hundreds of alerts per event | Raise `alerts.cooldown_sec` → `5–10` |
| Camera keeps disconnecting | Set `camera.transport: tcp`, lower `reconnect_delay_sec` |
| High CPU despite CUDA | Verify `gpu.enabled: true` and CUDA OpenCV build |
| NVDEC decode fails silently | Check FFmpeg NVDEC support — stream falls back to CPU automatically |

---

## 📦 Requirements

```
opencv-python>=4.8.0     # or opencv-contrib-python for GPU build
PyYAML>=6.0
```

---

<div align="center">

Built for production surveillance deployments · GPU-first · Config-driven · Zero code changes to deploy

</div>