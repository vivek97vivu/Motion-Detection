"""
utils/config_loader.py
──────────────────────
Loads config/config.yaml and exposes a typed, dot-accessible ConfigBox.
Validates required keys on startup so misconfiguration is caught early.
"""

import os
import yaml
from pathlib import Path
from typing import Any


# ── Minimal dot-access wrapper ────────────────────────────────────────────────
class ConfigBox(dict):
    """A dict subclass that supports attribute access: cfg.camera.rtsp_url"""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config key '{key}' not found.")
        return ConfigBox(val) if isinstance(val, dict) else val

    def __setattr__(self, key, value):
        self[key] = value

    def get_list(self, key: str, default=None):
        val = self.get(key, default)
        return list(val) if val is not None else default


# ── Loader ────────────────────────────────────────────────────────────────────


_REQUIRED_KEYS = [
    ("camera", "rtsp_url"),
    ("camera", "transport"),
    ("detection", "min_contour_area_px2"),

    ("alerts", "snapshot_dir"),
    ("alerts", "json_dir"),

    ("alerts", "cooldown_sec"),
    ("logging", "config_file"),
]


def load_config(config_path: str = "config/config.yaml") -> ConfigBox:
    """
    Load and return the main config as a ConfigBox.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (relative to the project root).

    Raises
    ------
    FileNotFoundError  – config file missing
    KeyError           – a required key is absent
    ValueError         – invalid value (e.g. bad transport)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}\n"
            "Copy config/config.yaml from the project template and edit it."
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = ConfigBox(raw)

    # ── validate required keys ────────────────────────────────────────────────
    for section, key in _REQUIRED_KEYS:
        if section not in cfg or key not in cfg[section]:
            raise KeyError(
                f"Missing required config key: [{section}] → {key}\n"
                f"Check {config_path}"
            )

    # ── validate allowed values ───────────────────────────────────────────────
    transport = cfg["camera"]["transport"].lower()
    if transport not in ("tcp", "udp"):
        raise ValueError(
            f"camera.transport must be 'tcp' or 'udp', got '{transport}'"
        )

    quality = cfg["alerts"]["jpeg_quality"]
    if not (1 <= int(quality) <= 100):
        raise ValueError(
            f"alerts.jpeg_quality must be 1–100, got {quality}"
        )

    return cfg
