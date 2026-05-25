

import json
import os
import datetime
import time

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("alert")


class AlertManager:

    def __init__(self, cfg):
        al = cfg.alerts
        self._snapshot_dir     = al.snapshot_dir
        self._json_dir         = al.json_dir
        self._cooldown         = float(al.cooldown_sec)
        self._jpeg_quality     = int(al.jpeg_quality)
        self._draw_boxes       = bool(al.draw_bounding_boxes)
        self._draw_zone        = bool(al.draw_zone_on_snapshot)
        self._watermark        = bool(al.timestamp_watermark)
        self._camera_url       = cfg.camera.rtsp_url

        # display config for annotation colours
        disp = cfg.display
        self._box_color  = tuple(disp.get_list("box_color",  [0, 0, 255]))
        self._zone_color = tuple(disp.get_list("zone_color_normal", [0, 220, 0]))

        self._last_alert_time = 0.0
        self._alert_id        = 0

        os.makedirs(self._snapshot_dir, exist_ok=True)
        os.makedirs(self._json_dir, exist_ok=True)
        
        log.info(
            "AlertManager ready  snapshot_dir=%s  json_dir=%s  cooldown=%.1fs",
            os.path.abspath(self._snapshot_dir), os.path.abspath(self._json_dir), self._cooldown,
        )

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def alert_count(self) -> int:
        return self._alert_id

    def cooldown_active(self) -> bool:
        return (time.time() - self._last_alert_time) < self._cooldown

    def save(
        self,
        frame: np.ndarray,
        roi: tuple | None,
        contours: list,
        force: bool = False,
    ) -> tuple[str, str] | None:
        """
        Save a snapshot + JSON file for a motion event.

        Parameters
        ----------
        frame    : Full BGR frame from the camera.
        roi      : (x, y, w, h) detection zone or None.
        contours : Significant contours in full-frame coordinates.
        force    : If True, bypass the cooldown timer.

        Returns
        -------
        (img_path, json_path)  or  None if cooldown is active.
        """
        if not force and self.cooldown_active():
            return None

        self._alert_id    += 1
        self._last_alert_time = time.time()

        ts     = datetime.datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")


        img_path = os.path.join(
            self._snapshot_dir,
            f"alert_{ts_str}.jpg"
        )

        json_path = os.path.join(
            self._json_dir,
            f"alert_{ts_str}.json"
        )

        # ── build annotated image ─────────────────────────────────────────────
        annotated = frame.copy()
        boxes     = []

        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            area = int(cv2.contourArea(c))
            boxes.append({
                "x": cx, "y": cy, "width": cw, "height": ch,
                "area_px2": area,
            })
            if self._draw_boxes:
                cv2.rectangle(
                    annotated, (cx, cy), (cx + cw, cy + ch),
                    self._box_color, 2,
                )

        if self._draw_zone and roi:
            rx, ry, rw, rh = roi
            cv2.rectangle(
                annotated, (rx, ry), (rx + rw, ry + rh),
                self._zone_color, 2,
            )

        if self._watermark:
            label = ts.strftime("Captured  %Y-%m-%d  %H:%M:%S")
            cv2.putText(
                annotated, label,
                (8, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1,
            )

        cv2.imwrite(img_path, annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

        # ── build JSON metadata ───────────────────────────────────────────────
        alert_data = {
            "alert_id":    self._alert_id,
            "timestamp":   ts.isoformat(),
            "camera_url":  self._camera_url,
            "snapshot_file": os.path.abspath(img_path),
            "detection_zone": (
                {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
                if roi else None
            ),
            "motion_objects":        boxes,
            "object_count":          len(boxes),
            "total_motion_area_px2": sum(b["area_px2"] for b in boxes),
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(alert_data, fh, indent=2)

        # structured log entry (lands in logs/alerts.log as JSON)
        log.info(
            "Alert saved",
            extra={
                "alert_id":     self._alert_id,
                "snapshot":     img_path,
                "object_count": len(boxes),
                "total_area":   alert_data["total_motion_area_px2"],
            },
        )

        return img_path, json_path
