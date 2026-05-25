"""
core/detector.py
────────────────
Reference-frame motion detector.

How it works
────────────
1. When the ROI is drawn (or redrawn), the very next clean frame inside
   that zone is captured as the REFERENCE frame and frozen.
2. Every subsequent live frame is diff-ed against that frozen reference
   using absolute difference + threshold — NOT a learning background model.
3. Any pixel that changed by more than `diff_threshold` is marked as
   foreground.  Contours are found on that binary mask.

This means:
  • The detector never adapts — if something enters the zone it stays
    detected for as long as it differs from the reference.
  • Works perfectly for high/fixed-mount cameras watching an empty zone.
  • GPU-accelerated via cv2.UMat (OpenCL) — all heavy ops run on the GPU;
    only findContours pulls data back to CPU (no GPU version exists).

Parameters (config.detection)
──────────────────────────────
  diff_threshold        – per-pixel absolute difference to call "changed"
                          (replaces mog2_threshold; good default: 25)
  min_contour_area_px2  – blobs smaller than this are noise-filtered
  blur_kernel_size      – Gaussian blur before diff (reduces noise)
  dilate_iterations     – dilate foreground mask to fill object gaps
"""

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("detector")


class MotionDetector:
    """
    Reference-frame diff detector with OpenCL (UMat) GPU acceleration.

    Parameters
    ----------
    cfg : ConfigBox
        Full application config — uses cfg.detection section.
    """

    def __init__(self, cfg):
        det = cfg.detection
        self._min_area      = int(det.min_contour_area_px2)
        self._blur_ksize    = int(det.blur_kernel_size)
        if self._blur_ksize % 2 == 0:
            self._blur_ksize += 1
        self._dilate_iters  = int(det.dilate_iterations)

        # diff_threshold: how different a pixel must be to count as motion
        # Falls back to mog2_threshold if diff_threshold not in config
        self._diff_thresh = int(
            det.get("diff_threshold", det.get("mog2_threshold", 25))
        )

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

        # ── Reference frame (frozen when ROI is locked) ───────────────────
        self._reference_gray: cv2.UMat | None = None   # GPU UMat
        self._reference_roi:  tuple | None     = None  # (x,y,w,h) it was built for
        self._ref_locked: bool                 = False

        # OpenCL availability
        _ocl = cv2.ocl.haveOpenCL()
        if _ocl:
            cv2.ocl.setUseOpenCL(True)
        log.info(
            "MotionDetector ready  mode=reference_diff  "
            "diff_thresh=%d  min_area=%dpx²  blur=%d  OpenCL=%s",
            self._diff_thresh, self._min_area, self._blur_ksize,
            "yes" if _ocl else "no (CPU fallback)",
        )

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def reference_locked(self) -> bool:
        return self._ref_locked

    def lock_reference(self, frame: np.ndarray, roi: tuple | None):
        """
        Capture and freeze the reference frame for the given ROI.

        Call this once after the user draws / confirms the detection zone.
        The detector will compare every future frame against this snapshot.

        Parameters
        ----------
        frame : np.ndarray   Clean BGR frame (should be the first live frame).
        roi   : tuple|None   (x, y, w, h) or None for full frame.
        """
        region = self._crop_roi(frame, roi)
        gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray, (self._blur_ksize, self._blur_ksize), 0
        )
        # Upload to GPU memory
        self._reference_gray = cv2.UMat(blurred)
        self._reference_roi  = roi
        self._ref_locked     = True
        log.info(
            "Reference frame locked  roi=%s",
            f"x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}" if roi else "full frame",
        )

    def detect(self, frame: np.ndarray, roi: tuple | None):
        """
        Compare *frame* against the locked reference frame inside *roi*.

        If no reference has been locked yet, locks the current frame
        automatically and returns (False, [], None).

        Parameters
        ----------
        frame : np.ndarray   Full BGR frame from the camera.
        roi   : tuple | None (x, y, w, h) or None for full frame.

        Returns
        -------
        motion_detected : bool
        contours        : list of significant contours (full-frame coordinates)
        fg_mask         : uint8 ndarray — binary foreground mask of the ROI
        """
        # Auto-lock on first call if caller didn't call lock_reference()
        if not self._ref_locked or self._reference_roi != roi:
            self.lock_reference(frame, roi)
            return False, [], None

        x_off, y_off = (roi[0], roi[1]) if roi else (0, 0)
        region = self._crop_roi(frame, roi)

        # ── GPU path (all UMat ops run on OpenCL device) ──────────────────
        gray_umat = cv2.cvtColor(cv2.UMat(region), cv2.COLOR_BGR2GRAY)
        blurred_umat = cv2.GaussianBlur(
            gray_umat, (self._blur_ksize, self._blur_ksize), 0
        )

        # Absolute diff against frozen reference
        diff_umat = cv2.absdiff(self._reference_gray, blurred_umat)

        # Threshold → binary mask
        _, fg_umat = cv2.threshold(
            diff_umat, self._diff_thresh, 255, cv2.THRESH_BINARY
        )

        # Morphological open (remove noise) then dilate (fill gaps)
        fg_umat = cv2.morphologyEx(fg_umat, cv2.MORPH_OPEN,  self._morph_kernel)
        fg_umat = cv2.dilate(fg_umat, self._morph_kernel, iterations=self._dilate_iters)

        # Download to CPU only for contour finding (no GPU findContours)
        fg_mask = fg_umat.get()

        raw_contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter noise; shift coords back to full-frame space
        significant = []
        for c in raw_contours:
            if cv2.contourArea(c) >= self._min_area:
                significant.append(c + [[[x_off, y_off]]])

        motion = len(significant) > 0
        if motion:
            log.debug("Motion detected  objects=%d", len(significant))

        return motion, significant, fg_mask

    def reset(self, frame: np.ndarray | None = None, roi: tuple | None = None):
        """
        Re-lock the reference frame (call after ROI is redrawn).

        If *frame* is provided, locks it immediately.
        Otherwise clears the lock so the next detect() call will auto-lock.
        """
        self._ref_locked     = False
        self._reference_gray = None
        self._reference_roi  = None

        if frame is not None:
            self.lock_reference(frame, roi)
        else:
            log.info("Reference cleared — will lock on next frame.")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_roi(frame: np.ndarray, roi: tuple | None) -> np.ndarray:
        if roi:
            x, y, w, h = roi
            return frame[y: y + h, x: x + w]
        return frame