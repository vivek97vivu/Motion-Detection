# RTSP Motion Detector — Production Setup

## Project Structure

```
motion_detector/
│
├── main.py                   ← Entry point — run this
│
├── config/
│   ├── config.yaml           ← ALL settings live here (camera, detection, alerts, display)
│   └── logging.yaml          ← Log handlers, levels, rotation, file paths
│
├── core/
│   ├── stream.py             ← Threaded RTSP reader with auto-reconnect
│   ├── detector.py           ← MOG2 background subtraction + contour filtering
│   ├── alert_manager.py      ← Saves JPEG snapshots + JSON metadata
│   └── display.py            ← All OpenCV drawing / ROI selection
│
├── utils/
│   ├── config_loader.py      ← Loads & validates config.yaml
│   ├── logger.py             ← Sets up logging from logging.yaml + JSON formatter
│   └── __init__.py
│
├── alerts/                   ← Auto-created — snapshots + JSON alerts saved here
├── logs/                     ← Auto-created — rotating log files saved here
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

Edit **`config/config.yaml`** — set your RTSP URL:
```yaml
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1"
```

Run:
```bash
python main.py
```

---

## Configuration Guide

### `config/config.yaml` — Application Settings

| Section | Key | Default | Description |
|---|---|---|---|
| `camera` | `rtsp_url` | *(required)* | Full RTSP URL of your IP camera |
| `camera` | `transport` | `tcp` | `tcp` (reliable) or `udp` (low latency) |
| `camera` | `reconnect_delay_sec` | `5` | Seconds before reconnect on drop |
| `camera` | `connection_timeout_sec` | `30` | Max wait for first frame on startup |
| `detection` | `min_contour_area_px2` | `500` | Smaller blobs are ignored (raise to reduce false alarms) |
| `detection` | `mog2_history` | `500` | Frames for background model (raise for stable bg) |
| `detection` | `mog2_threshold` | `16` | Per-pixel threshold (lower = more sensitive) |
| `alerts` | `cooldown_sec` | `1.0` | Minimum gap between saved alerts |
| `alerts` | `jpeg_quality` | `92` | Snapshot quality (1–100) |
| `display` | `zone_color_normal` | `[0,220,0]` | BGR colour of zone when idle |
| `display` | `zone_color_motion` | `[0,0,255]` | BGR colour of zone when motion |

### `config/logging.yaml` — Log Settings

| Log file | Contents |
|---|---|
| `logs/app.log` | All application events (DEBUG+), rotated daily, 30-day retention |
| `logs/error.log` | WARNING and above only, rotating 10MB×5 |
| `logs/alerts.log` | JSON-formatted alert events only (machine-parseable) |
| `logs/stream.log` | RTSP connection / reconnect events, daily rotation |

---

## Alert Output

Every motion event writes two files to `alerts/`:

```
alerts/alert_20260525_143210_847123.jpg   ← annotated snapshot
alerts/alert_20260525_143210_847123.json  ← metadata
```

**JSON structure:**
```json
{
  "alert_id": 3,
  "timestamp": "2026-05-25T14:32:10.847123",
  "camera_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
  "snapshot_file": "/home/user/motion_detector/alerts/alert_....jpg",
  "detection_zone": { "x": 120, "y": 80, "width": 400, "height": 300 },
  "motion_objects": [
    { "x": 200, "y": 150, "width": 60, "height": 90, "area_px2": 5400 }
  ],
  "object_count": 1,
  "total_motion_area_px2": 5400
}
```

---

## Live Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit cleanly |
| `R` | Pause and redraw detection zone (resets background model) |
| `S` | Force-save a snapshot immediately (bypasses cooldown) |

---

## Tuning Tips

- **Too many false alerts?** → Raise `detection.min_contour_area_px2` (e.g. 1000–2000)
- **Missing slow-moving objects?** → Lower `detection.mog2_threshold` (e.g. 8–12)
- **Flickering detections on lighting changes?** → Raise `detection.mog2_history` (e.g. 1000)
- **Camera keeps dropping?** → Switch `camera.transport` to `tcp`, lower `reconnect_delay_sec`
- **Too many log files?** → Reduce `backupCount` in `config/logging.yaml`
