"""
core/display.py
───────────────
CPU optimisations vs original
──────────────────────────────
• render() draws directly onto the live frame instead of frame.copy().
  Caller passes frame in; we annotate it in-place — no memcpy.
• Status bar timestamp updated only once per second, not every frame.
• no_signal_frame() pre-allocates the placeholder once at init and
  returns a copy of it — avoids np.zeros() allocation every 200 ms.
"""

import datetime

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("display")


class Display:
    def __init__(self, cfg):
        disp = cfg.display
        self._title       = disp.window_title
        self._show_status = bool(disp.show_status_bar)
        self._show_banner = bool(disp.show_motion_banner)
        self._zone_normal = tuple(disp.get_list("zone_color_normal", [0, 220, 0]))
        self._zone_motion = tuple(disp.get_list("zone_color_motion", [0, 0, 255]))
        self._box_color   = tuple(disp.get_list("box_color",         [0, 0, 255]))

        # pre-build the no-signal placeholder once
        self._no_signal_reconnecting = self._build_no_signal(False)
        self._no_signal_buffering    = self._build_no_signal(True)

        # status bar cache — reformat string only once per second
        self._status_last_sec: int  = -1
        self._status_ts_str:   str  = ""

    # ── ROI selection ─────────────────────────────────────────────────────────

    def select_roi(self, frame: np.ndarray) -> tuple | None:
        clone   = frame.copy()
        h, w    = clone.shape[:2]
        overlay = clone.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, clone, 0.35, 0, clone)
        cv2.putText(
            clone,
            "Drag to draw detection zone  |  ENTER/SPACE = confirm  |  ESC = full frame",
            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 230), 2,
        )
        roi = cv2.selectROI(
            "[ Draw Detection Zone ]", clone,
            fromCenter=False, showCrosshair=True,
        )
        cv2.destroyWindow("[ Draw Detection Zone ]")
        x, y, rw, rh = roi
        if rw < 10 or rh < 10:
            log.info("ROI cancelled — using full frame.")
            return None
        log.info("ROI selected: x=%d y=%d w=%d h=%d", x, y, rw, rh)
        return (x, y, rw, rh)

    # ── live frame rendering ──────────────────────────────────────────────────

    def render(
        self,
        frame:       np.ndarray,
        roi:         tuple | None,
        contours:    list,
        motion:      bool,
        alert_count: int,
        sensitivity: int,
    ) -> np.ndarray:
        """
        Annotate *frame* in-place and return it.

        No copy is made — caller should NOT reuse frame after this call.
        (stream.read() returns a fresh copy each time, so this is safe.)
        """
        display = frame   # draw directly — no memcpy

        # bounding boxes
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            cv2.rectangle(display, (cx, cy), (cx + cw, cy + ch),
                          self._box_color, 2)

        # zone rectangle
        if roi:
            rx, ry, rw, rh = roi
            color = self._zone_motion if motion else self._zone_normal
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), color, 2)
            label = "!! MOTION DETECTED !!" if motion else "DETECTION ZONE"
            cv2.putText(display, label, (rx + 4, max(ry - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

        # motion banner
        if motion and self._show_banner:
            cv2.rectangle(display, (0, 0), (display.shape[1], 54), (0, 0, 160), -1)
            cv2.putText(display, "  *** MOTION DETECTED ***",
                        (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2)

        # status bar — reformat only once per second
        if self._show_status:
            now_sec = int(datetime.datetime.now().timestamp())
            if now_sec != self._status_last_sec:
                self._status_ts_str  = datetime.datetime.now().strftime(
                    "%Y-%m-%d  %H:%M:%S"
                )
                self._status_last_sec = now_sec

            bar = (
                f"{self._status_ts_str}   |   Alerts: {alert_count}"
                f"   |   Sensitivity: {sensitivity} px²"
                f"   |   [Q]uit  [R]edraw  [S]ave"
            )
            cv2.putText(display, bar, (8, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 180, 180), 1)

        return display

    def show(self, frame: np.ndarray):
        cv2.imshow(self._title, frame)

    def wait_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def destroy(self):
        cv2.destroyAllWindows()

    # ── no-signal placeholder ─────────────────────────────────────────────────

    def no_signal_frame(self, connecting: bool) -> np.ndarray:
        """Return a pre-built placeholder (copy, so caller can annotate safely)."""
        src = self._no_signal_buffering if connecting else self._no_signal_reconnecting
        return src.copy()

    @staticmethod
    def _build_no_signal(buffering: bool) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype="uint8")
        msg = "Buffering…" if buffering else "Reconnecting to camera…"
        cv2.putText(img, msg, (640 // 2 - 160, 480 // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 100, 100), 2)
        cv2.putText(img, "Press Q to quit",
                    (640 // 2 - 90, 480 // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        return img