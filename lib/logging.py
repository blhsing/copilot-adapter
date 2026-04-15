"""Custom logging utilities."""

import copy
import logging
from pathlib import Path

_NOISY_MESSAGES = frozenset({
    "Waiting for application startup.",
    "Application startup complete.",
    "Waiting for application shutdown.",
    "Application shutdown complete.",
    "Started server process [%d]",
})


class LifespanFilter(logging.Filter):
    """Filter out repetitive per-worker uvicorn lifespan messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.msg not in _NOISY_MESSAGES


def build_logging_config(base_config: dict, log_level: str, log_file: str | None = None) -> dict:
    """Return a Uvicorn logging config with optional additive file logging."""
    config = copy.deepcopy(base_config)
    config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelprefix)s %(message)s"
    config["formatters"]["access"]["fmt"] = (
        '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )

    config.setdefault("filters", {})
    config["filters"]["no_lifespan"] = {"()": "lib.logging.LifespanFilter"}

    config.setdefault("handlers", {})
    config["handlers"].setdefault("default", {}).setdefault("filters", [])
    if "no_lifespan" not in config["handlers"]["default"]["filters"]:
        config["handlers"]["default"]["filters"].append("no_lifespan")

    handler_names = ["default"]
    access_handler_names = ["access"]

    if log_file:
        log_path = Path(log_file)
        if log_path.parent != Path("."):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(log_path),
            "encoding": "utf-8",
            "mode": "a",
            "filters": ["no_lifespan"],
        }
        config["handlers"]["access_file"] = {
            "class": "logging.FileHandler",
            "formatter": "access",
            "filename": str(log_path),
            "encoding": "utf-8",
            "mode": "a",
            "filters": ["no_lifespan"],
        }
        handler_names.append("file")
        access_handler_names.append("access_file")

    config.setdefault("loggers", {})
    config["loggers"]["lib"] = {
        "handlers": handler_names,
        "level": log_level.upper(),
        "propagate": False,
    }
    config["loggers"]["uvicorn"] = {
        **config["loggers"].get("uvicorn", {}),
        "handlers": handler_names,
        "level": log_level.upper(),
        "propagate": False,
    }
    config["loggers"]["uvicorn.error"] = {
        **config["loggers"].get("uvicorn.error", {}),
        "handlers": handler_names,
        "level": log_level.upper(),
        "propagate": False,
    }
    config["loggers"]["uvicorn.access"] = {
        **config["loggers"].get("uvicorn.access", {}),
        "handlers": access_handler_names,
        "level": log_level.upper(),
        "propagate": False,
    }
    return config


def resolve_worker_log_file(default: str | None = None) -> str | None:
    """Read the inherited worker log-file override from the environment."""
    from os import environ

    return environ.get("_COPILOT_ADAPTER_LOG_FILE", default)


def build_runtime_logging_config(base_config: dict, log_level: str, log_file: str | None = None) -> dict:
    """Build logging config, honoring the worker env override when present."""
    return build_logging_config(base_config, log_level, resolve_worker_log_file(log_file))


__all__ = ["LifespanFilter", "build_logging_config", "build_runtime_logging_config", "resolve_worker_log_file"]
