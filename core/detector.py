"""
core/detector.py
────────────────
Reference-frame diff detector — GPU via cv2.UMat (OpenCL).

CPU optimisations vs original
──────────────────────────────
• ROI is cropped BEFORE blur/diff — only zone pixels are ever processed.
• No frame.copy() inside detect() — we slice the ROI directly.
• reference_gray is stored as UMat — stays in GPU memory between frames.
• All heavy ops (blur, absdiff, threshold, morph, dilate) run on UMat.
• Only findContours pulls to CPU (no GPU version exists in OpenCV).
"""

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("detector")


class MotionDetector:
    def __init__(self, cfg):
        det = cfg.detection
        self._min_area     = int(det.min_contour_area_px2)
        self._blur_ksize   = int(det.blur_kernel_size)
        if self._blur_ksize % 2 == 0:
            self._blur_ksize += 1
        self._dilate_iters = int(det.dilate_iterations)
        self._diff_thresh  = int(
            det.get("diff_threshold", det.get("mog2_threshold", 25))
        )

        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._reference_gray: cv2.UMat | None = None
        self._reference_roi:  tuple | None    = None
        self._ref_locked: bool                = False

        _ocl = cv2.ocl.haveOpenCL()
        if _ocl:
            cv2.ocl.setUseOpenCL(True)

        log.info(
            "MotionDetector  mode=reference_diff  diff_thresh=%d  "
            "min_area=%dpx²  blur=%d  OpenCL=%s",
            self._diff_thresh, self._min_area, self._blur_ksize,
            "yes" if _ocl else "no",
        )

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def reference_locked(self) -> bool:
        return self._ref_locked

    def lock_reference(self, frame: np.ndarray, roi: tuple | None):
        """Freeze the current ROI region as the reference for future diffs."""
        # Crop first — only process the zone, not the whole frame
        region  = self._crop(frame, roi)
        gray    = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray, (self._blur_ksize, self._blur_ksize), 0
        )
        # Upload to GPU memory — stays there until reset()
        self._reference_gray = cv2.UMat(blurred)
        self._reference_roi  = roi
        self._ref_locked     = True
        log.info(
            "Reference locked  roi=%s",
            f"x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}" if roi else "full frame",
        )

    def detect(self, frame: np.ndarray, roi: tuple | None):
        """
        Diff *frame* ROI against the locked reference.

        Returns (motion_bool, contours_list, fg_mask_ndarray).
        Auto-locks on first call if not already locked.
        """
        if not self._ref_locked or self._reference_roi != roi:
            self.lock_reference(frame, roi)
            return False, [], None

        x_off = roi[0] if roi else 0
        y_off = roi[1] if roi else 0

        # ── crop first — all expensive ops only touch the ROI pixels ─────────
        region = self._crop(frame, roi)   # no copy; just a numpy slice view

        # ── GPU path (UMat) ───────────────────────────────────────────────────
        gray_u    = cv2.cvtColor(cv2.UMat(region), cv2.COLOR_BGR2GRAY)
        blurred_u = cv2.GaussianBlur(
            gray_u, (self._blur_ksize, self._blur_ksize), 0
        )
        diff_u    = cv2.absdiff(self._reference_gray, blurred_u)
        _, fg_u   = cv2.threshold(diff_u, self._diff_thresh, 255, cv2.THRESH_BINARY)
        fg_u      = cv2.morphologyEx(fg_u, cv2.MORPH_OPEN,   self._morph_kernel)
        fg_u      = cv2.dilate(fg_u, self._morph_kernel, iterations=self._dilate_iters)

        # Download only the small ROI mask (not the full frame)
        fg_mask = fg_u.get()

        raw, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        significant = [
            c + [[[x_off, y_off]]]
            for c in raw
            if cv2.contourArea(c) >= self._min_area
        ]

        motion = bool(significant)
        if motion:
            log.debug("Motion  objects=%d", len(significant))

        return motion, significant, fg_mask

    def reset(self, frame: np.ndarray | None = None, roi: tuple | None = None):
        """Clear the reference. Optionally re-lock immediately if frame given."""
        self._ref_locked     = False
        self._reference_gray = None
        self._reference_roi  = None
        if frame is not None:
            self.lock_reference(frame, roi)
        else:
            log.info("Reference cleared — will lock on next frame.")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _crop(frame: np.ndarray, roi: tuple | None) -> np.ndarray:
        """Return a view (not a copy) of the ROI region."""
        if roi:
            x, y, w, h = roi
            return frame[y: y + h, x: x + w]   # view, no memcpy
        return frame