from __future__ import annotations

import logging
from pathlib import Path

from src.preprocess import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "logs"
TRAIN_LOG_PATH = LOGS_DIR / "train.log"
APP_LOG_PATH = LOGS_DIR / "app.log"


def get_logger(name: str, log_path: Path) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_log_path = log_path.resolve()
    existing_handler = any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_log_path
        for handler in logger.handlers
    )

    if not existing_handler:
        file_handler = logging.FileHandler(resolved_log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(file_handler)

    return logger


def get_train_logger() -> logging.Logger:
    return get_logger("nexhealth.train", TRAIN_LOG_PATH)


def get_app_logger() -> logging.Logger:
    return get_logger("nexhealth.app", APP_LOG_PATH)
