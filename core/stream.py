"""
core/stream.py
──────────────
Opens and maintains an RTSP camera stream in a background thread.
Uses GStreamer + nvh265dec for GPU-accelerated decoding.
Falls back to FFmpeg automatically if GStreamer is unavailable.
"""

import os
import threading
import time

import cv2

from utils.logger import get_logger

log = get_logger("stream")


def _build_gst_pipeline(url: str, transport: str) -> str:
    """Return a GStreamer pipeline string for the given RTSP URL."""
    return (
        f"rtspsrc location={url} protocols={transport} "
        f"latency=0 drop-on-latency=true "
        f"! rtph265depay ! h265parse ! nvh265dec "
        f"! videoconvert ! video/x-raw,format=BGR "
        f"! appsink drop=true max-buffers=1 sync=false"
    )


class RTSPStream:
    """
    Opens and maintains an RTSP camera stream in a background thread.

    Decoding priority:
      1. GStreamer + nvh265dec  (GPU, zero-copy)
      2. FFmpeg                 (CPU fallback)

    Parameters
    ----------
    cfg : ConfigBox
        Full application config (uses cfg.camera section).
    """

    def __init__(self, cfg):
        cam = cfg.camera
        self._url             = cam.rtsp_url
        self._transport       = cam.transport.lower()
        self._reconnect_delay = int(cam.reconnect_delay_sec)
        self._buf_size        = int(cam.frame_buffer_size)

        self._cap       = None
        self._frame     = None
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._connected = False
        self._using_gst = False          # set after first successful open

        self._thread = threading.Thread(
            target=self._reader_loop, name="rtsp-reader", daemon=True
        )
        self._thread.start()
        log.info("RTSPStream started  url=%s  transport=%s",
                 self._url, self._transport)

    # ── public API ────────────────────────────────────────────────────────────

    def read(self):
        """Return ``(True, frame)`` or ``(False, None)`` when no frame yet."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def decoder(self) -> str:
        """Human-readable label for the active decoder."""
        return "GStreamer/nvh265dec (GPU)" if self._using_gst else "FFmpeg (CPU)"

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        log.info("Stopping RTSPStream…")
        self._stop_evt.set()
        self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        log.info("RTSPStream stopped.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _open_gstreamer(self):
        """Try to open the stream via GStreamer + nvh265dec."""
        pipeline = _build_gst_pipeline(self._url, self._transport)
        log.debug("Trying GStreamer pipeline: %s", pipeline)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self._using_gst = True
            log.info("✅ GStreamer/nvh265dec pipeline active (GPU decode)")
            return cap
        cap.release()
        return None

    def _open_ffmpeg(self):
        """Open the stream via OpenCV/FFmpeg (CPU fallback)."""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self._transport}|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "max_delay;500000"
        )
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buf_size)
        if cap.isOpened():
            self._using_gst = False
            log.warning("⚠️  GStreamer unavailable — falling back to FFmpeg (CPU decode)")
            return cap
        cap.release()
        return None

    def _open_capture(self):
        """Try GStreamer first, fall back to FFmpeg."""
        if self._cap:
            self._cap.release()
        cap = self._open_gstreamer()
        if cap is None:
            cap = self._open_ffmpeg()
        return cap

    def _reader_loop(self):
        """Background thread: grab frames; reconnect silently on failure."""
        while not self._stop_evt.is_set():
            log.debug("Opening RTSP connection…")
            self._cap = self._open_capture()

            if self._cap is None or not self._cap.isOpened():
                log.warning("Cannot connect to camera. Retrying in %ds…",
                            self._reconnect_delay)
                self._connected = False
                time.sleep(self._reconnect_delay)
                continue

            log.info("Camera connected: %s  decoder=%s", self._url, self.decoder)
            self._connected = True

            consecutive_failures = 0
            while not self._stop_evt.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    consecutive_failures += 1
                    log.warning(
                        "Frame read failed (attempt %d). Reconnecting in %ds…",
                        consecutive_failures, self._reconnect_delay,
                    )
                    self._connected = False
                    time.sleep(self._reconnect_delay)
                    break

                consecutive_failures = 0
                with self._lock:
                    self._frame = frame