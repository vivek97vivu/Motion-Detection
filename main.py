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

    # ── 1. Load config ────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── 2. Set up logging ─────────────────────────────────────────────────────
    setup_logging(cfg.logging.config_file)

    log.info("=" * 60)
    log.info("Motion Detector starting")
    log.info("Config    : %s", args.config)
    log.info("Camera    : %s", cfg.camera.rtsp_url)
    log.info("Snapshots : %s", cfg.alerts.snapshot_dir)
    log.info("JSON      : %s", cfg.alerts.json_dir)
    log.info("=" * 60)

    # ── 3. Initialise components ──────────────────────────────────────────────
    stream   = RTSPStream(cfg)
    detector = MotionDetector(cfg)
    alertmgr = AlertManager(cfg)
    display  = Display(cfg)

    # ── 4. Wait for first frame ───────────────────────────────────────────────
    print("Connecting to camera… (Ctrl-C to abort)")
    timeout = time.time() + int(cfg.camera.connection_timeout_sec)
    while True:
        ok, frame = stream.read()
        if ok and frame is not None:
            break
        if time.time() > timeout:
            log.error(
                "No frame received within %ds. Check RTSP URL and network.",
                cfg.camera.connection_timeout_sec,
            )
            stream.stop()
            sys.exit(1)
        time.sleep(0.2)

    log.info("First frame received  decoder=%s", stream.decoder)
    print(f"\nFirst frame received ({stream.decoder}) — draw your detection zone.")

    # ── 5. ROI selection + reference frame lock ───────────────────────────────
    roi = display.select_roi(frame)
    if roi is None:
        log.info("No ROI drawn — monitoring full frame.")
    else:
        log.info("Detection zone: x=%d y=%d w=%d h=%d", *roi)

    # ★ Lock the reference frame immediately after ROI is confirmed.
    #   The detector will diff every live frame against this frozen snapshot.
    detector.lock_reference(frame, roi)
    log.info("Reference frame locked.  Detector mode: reference-diff (GPU)")

    print(
        "\n=== Running ===\n"
        "  Q / ESC  → quit\n"
        "  R        → redraw detection zone  (re-locks reference)\n"
        "  S        → force-save snapshot now\n"
    )

    # ── 6. Main loop ──────────────────────────────────────────────────────────
    significant = []

    while True:
        ok, frame = stream.read()

        # no-signal / reconnecting screen
        if not ok or frame is None:
            placeholder = Display.no_signal_frame(stream.is_connected)
            display.show(placeholder)
            key = display.wait_key(200)
            if key in (ord("q"), 27):
                break
            continue

        # motion analysis — diff against the locked reference frame
        motion, significant, _mask = detector.detect(frame, roi)

        # save alert (AlertManager handles cooldown)
        if motion:
            alertmgr.save(frame, roi, significant)

        # render and display
        rendered = display.render(
            frame       = frame,
            roi         = roi,
            contours    = significant,
            motion      = motion,
            alert_count = alertmgr.alert_count,
            sensitivity = int(cfg.detection.min_contour_area_px2),
        )
        display.show(rendered)

        # keyboard handling
        key = display.wait_key(1)

        if key in (ord("q"), 27):
            log.info("Quit key pressed.")
            break

        elif key == ord("r"):
            # Redraw zone and immediately re-lock a fresh reference
            ok2, snap = stream.read()
            if ok2 and snap is not None:
                roi = display.select_roi(snap)
                detector.reset(frame=snap, roi=roi)   # ★ re-lock reference
                log.info(
                    "ROI updated + reference re-locked: %s",
                    f"x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}"
                    if roi else "full frame",
                )

        elif key == ord("s"):
            paths = alertmgr.save(frame, roi, significant, force=True)
            if paths:
                log.info("Forced snapshot saved: %s", paths[0])

    # ── 7. Cleanup ────────────────────────────────────────────────────────────
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