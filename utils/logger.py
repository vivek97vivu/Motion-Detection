"""
utils/logger.py
───────────────
Sets up Python logging from config/logging.yaml.
Creates the logs/ directory if it doesn't exist.
Call setup_logging() once at startup, then use get_logger() everywhere.
"""

import json
import logging
import logging.config
import logging.handlers
import os
import yaml
from datetime import datetime
from pathlib import Path


def setup_logging(logging_config_path: str = "config/logging.yaml") -> None:
    """
    Configure the logging system from the YAML file.
    Must be called once before any logger is used.

    Parameters
    ----------
    logging_config_path : str
        Path to logging.yaml (relative to the project root).
    """
    # ensure logs directory exists before handlers try to open files
    Path("logs").mkdir(exist_ok=True)

    path = Path(logging_config_path)
    if not path.exists():
        # fall back to basic console logging if YAML is missing
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.warning(
            "Logging config not found at %s — using basic console logging.",
            path.resolve(),
        )
        return

    with open(path, "r", encoding="utf-8") as fh:
        log_cfg = yaml.safe_load(fh)

    logging.config.dictConfig(log_cfg)
    logging.getLogger("motion_detector").debug(
        "Logging initialised from %s", path.resolve()
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger under the motion_detector hierarchy.

    Usage
    -----
        log = get_logger("stream")     # → motion_detector.stream
        log = get_logger("detector")   # → motion_detector.detector
        log = get_logger("alert")      # → motion_detector.alert
    """
    return logging.getLogger(f"motion_detector.{name}")


# ── JSON log formatter ────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """
    Emits each log record as a single JSON line.
    Used by the alert_file handler so alerts.log is machine-parseable.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        # attach any extra fields added via logger.info("…", extra={…})
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                payload[key] = val

        return json.dumps(payload, default=str)
