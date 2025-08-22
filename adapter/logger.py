"""
used to get logger
"""

import logging
import logging.handlers
from pathlib import Path

try:
    import colorlog  # optional
except Exception:  # pragma: no cover
    colorlog = None


def get_logger(name: str, path: Path):
    assert name is not None, "need a name for logger"
    # Ensure parent directory exists
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # avoid duplicate logs
        log_handler = logging.FileHandler(filename=path, mode='w')  # overwrite existing
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        logger.addHandler(log_handler)

        # Optional colorful stdout handler
        try:
            if colorlog is not None:
                log_handler_std = colorlog.StreamHandler()
                log_handler_std.setLevel(logging.INFO)
                log_handler_std.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(message)s"))
                # Disabled by default to keep console clean; uncomment to enable
                # logger.addHandler(log_handler_std)
        except Exception:
            pass

    return logger