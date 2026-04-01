"""Custom logging utilities."""

import logging

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
