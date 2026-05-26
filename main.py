"""
main.py
───────
Entry point. Changes vs original:
  • frame_interval sleep in main loop  → eliminates spin-burn
  • process_every_n_frames config      → skip frames between detections
  • lock_reference() called after ROI  → reference-frame diff detector
"""

import argparse
import sys
import time

from utils import load_config, setup_logging, get_logger
from core  import RTSPStream, MotionDetector, AlertManager, Display

log = get_logger("app")


def parse_args():
    p = argparse.ArgumentParser(description="Production RTSP Motion Detector")
    p.add_argument(
        "--config", default="config/config.yaml",
        help="Path to main config YAML (default: config/config.yaml)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.config_file)

    log.info("=" * 60)
    log.info("Motion Detector starting")
    log.info("Config    : %s", args.config)
    log.info("Camera    : %s", cfg.camera.rtsp_url)
    log.info("Snapshots : %s", cfg.alerts.snapshot_dir)
    log.info("JSON      : %s", cfg.alerts.json_dir)
    log.info("=" * 60)

    stream   = RTSPStream(cfg)
    detector = MotionDetector(cfg)
    alertmgr = AlertManager(cfg)
    display  = Display(cfg)

    # ── target frame interval (seconds) ──────────────────────────────────────
    # e.g. display_fps: 15 → sleep 1/15 = 66ms between iterations
    display_fps      = int(cfg.display.get("display_fps", 15))
    frame_interval   = 1.0 / display_fps

    # how many display frames to skip between detector runs
    # process_every_n: 2 means run detector on every 2nd frame
    detect_every_n   = int(cfg.detection.get("process_every_n_frames", 2))
    detect_counter   = 0

    # ── wait for first frame ──────────────────────────────────────────────────
    print("Connecting to camera… (Ctrl-C to abort)")
    timeout = time.time() + int(cfg.camera.connection_timeout_sec)
    while True:
        ok, frame = stream.read()
        if ok and frame is not None:
            break
        if time.time() > timeout:
            log.error("No frame received within %ds. Check RTSP URL.",
                      cfg.camera.connection_timeout_sec)
            stream.stop()
            sys.exit(1)
        time.sleep(0.2)

    log.info("First frame received  decoder=%s", stream.decoder)
    print(f"\nFirst frame received ({stream.decoder}) — draw your detection zone.")

    # ── ROI + reference frame lock ────────────────────────────────────────────
    roi = display.select_roi(frame)
    if roi is None:
        log.info("No ROI drawn — monitoring full frame.")
    else:
        log.info("Detection zone: x=%d y=%d w=%d h=%d", *roi)

    detector.lock_reference(frame, roi)
    log.info("Reference frame locked.  Detector: reference-diff (GPU UMat)")

    print(
        "\n=== Running ===\n"
        "  Q / ESC  → quit\n"
        "  R        → redraw detection zone\n"
        "  S        → force-save snapshot\n"
    )

    significant = []
    motion      = False
    last_time   = time.monotonic()

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        loop_start = time.monotonic()

        ok, frame = stream.read()

        if not ok or frame is None:
            placeholder = Display.no_signal_frame(stream.is_connected)
            display.show(placeholder)
            key = display.wait_key(200)
            if key in (ord("q"), 27):
                break
            continue

        # run detector only every N frames
        detect_counter += 1
        if detect_counter >= detect_every_n:
            detect_counter = 0
            motion, significant, _mask = detector.detect(frame, roi)
            if motion:
                alertmgr.save(frame, roi, significant)

        # render and display (every frame — keeps UI smooth)
        rendered = display.render(
            frame       = frame,
            roi         = roi,
            contours    = significant,
            motion      = motion,
            alert_count = alertmgr.alert_count,
            sensitivity = int(cfg.detection.min_contour_area_px2),
        )
        display.show(rendered)

        # keyboard
        key = display.wait_key(1)

        if key in (ord("q"), 27):
            log.info("Quit key pressed.")
            break

        elif key == ord("r"):
            ok2, snap = stream.read()
            if ok2 and snap is not None:
                roi = display.select_roi(snap)
                detector.reset(frame=snap, roi=roi)
                significant = []
                motion      = False
                log.info(
                    "ROI + reference updated: %s",
                    f"x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}"
                    if roi else "full frame",
                )

        elif key == ord("s"):
            paths = alertmgr.save(frame, roi, significant, force=True)
            if paths:
                log.info("Forced snapshot saved: %s", paths[0])

        # ── frame-rate cap: sleep the remainder of the frame interval ─────────
        elapsed = time.monotonic() - loop_start
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ── cleanup ───────────────────────────────────────────────────────────────
    stream.stop()
    display.destroy()
    log.info("Session ended. Total alerts: %d", alertmgr.alert_count)
    print(
        f"\nSession ended.\n"
        f"  Total alerts : {alertmgr.alert_count}\n"
        f"  Snapshots    : {cfg.alerts.snapshot_dir}/\n"
        f"  JSON files   : {cfg.alerts.json_dir}/\n"
        f"  Log files    : logs/\n"
    )


if __name__ == "__main__":
    main()