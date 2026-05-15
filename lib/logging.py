"""Custom logging utilities."""

import copy
import logging
import os
import socket
import threading
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


# --- Reverse-DNS cache and lookup ------------------------------------------

_MISSING = object()
_hostname_cache: dict[str, str | None] = {}
_hostname_lock = threading.Lock()
_lookup_events: dict[str, threading.Event] = {}


def _reverse_dns_server() -> str | None:
    return (
        os.environ.get("_COPILOT_ADAPTER_REVERSE_DNS_SERVER")
        or os.environ.get("COPILOT_ADAPTER_REVERSE_DNS_SERVER")
        or None
    )


def _reverse_dns_sync_wait_ms() -> int:
    raw = (
        os.environ.get("_COPILOT_ADAPTER_REVERSE_DNS_SYNC_WAIT_MS")
        or os.environ.get("COPILOT_ADAPTER_REVERSE_DNS_SYNC_WAIT_MS")
        or "0"
    )
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _lookup_hostname_sync(ip: str) -> str | None:
    dns_server = _reverse_dns_server()
    if dns_server:
        try:
            import dns.resolver
            import dns.reversename
        except ImportError:
            logging.getLogger(__name__).warning(
                "dnspython not installed; falling back to system resolver for %s", ip
            )
        else:
            try:
                resolver = dns.resolver.Resolver(configure=False)
                resolver.nameservers = [dns_server]
                resolver.lifetime = 2.0
                resolver.timeout = 2.0
                rev = dns.reversename.from_address(ip)
                answers = resolver.resolve(rev, "PTR")
                if answers:
                    return str(answers[0]).rstrip(".")
            except Exception:
                return None
            return None
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return None


def _schedule_lookup(ip: str) -> threading.Event:
    """Spawn a background lookup for *ip*. Caller holds ``_hostname_lock``
    and must register the returned event before releasing it."""
    event = threading.Event()
    _lookup_events[ip] = event

    def worker():
        name = _lookup_hostname_sync(ip)
        with _hostname_lock:
            _hostname_cache[ip] = name
            _lookup_events.pop(ip, None)
        event.set()

    threading.Thread(target=worker, daemon=True).start()
    return event


def get_cached_hostname(ip: str, sync_wait_ms: int | None = None) -> str | None:
    """Return a cached hostname for ``ip``.

    On a fresh cache miss, schedules a background DNS lookup. If
    ``sync_wait_ms`` (or the ``COPILOT_ADAPTER_REVERSE_DNS_SYNC_WAIT_MS``
    env override) is positive, blocks up to that many milliseconds for the
    lookup to complete so the very first log line for an IP can carry the
    hostname. Returns ``None`` if the lookup hasn't completed by then.
    """
    if not ip:
        return None
    if sync_wait_ms is None:
        sync_wait_ms = _reverse_dns_sync_wait_ms()
    event_to_wait: threading.Event | None = None
    with _hostname_lock:
        cached = _hostname_cache.get(ip, _MISSING)
        if cached is _MISSING:
            _hostname_cache[ip] = None  # placeholder so we only fire once
            event_to_wait = _schedule_lookup(ip)
        elif cached is None:
            # A prior caller scheduled a lookup that may still be in flight.
            event_to_wait = _lookup_events.get(ip)
        else:
            return cached
    if event_to_wait is not None and sync_wait_ms > 0:
        if event_to_wait.wait(sync_wait_ms / 1000.0):
            with _hostname_lock:
                return _hostname_cache.get(ip)
    return None


def _split_client_addr(client_addr: str) -> tuple[str, str | None]:
    """Return (ip, port) from a "host:port" string; port may be None."""
    if not client_addr:
        return "", None
    if client_addr.startswith("["):
        # IPv6 bracketed form: [::1]:1234
        rb = client_addr.rfind("]")
        if rb != -1:
            ip = client_addr[1:rb]
            rest = client_addr[rb + 1:]
            port = rest.lstrip(":") or None
            return ip, port
    ip, sep, port = client_addr.rpartition(":")
    if not sep:
        return client_addr, None
    return ip, port or None


def _build_access_formatter_class():
    """Lazily import uvicorn and return a hostname-aware access formatter."""
    from uvicorn.logging import AccessFormatter

    class HostnameAccessFormatter(AccessFormatter):
        def formatMessage(self, record: logging.LogRecord) -> str:
            args = record.args
            if isinstance(args, tuple) and args:
                client_addr = args[0]
                if isinstance(client_addr, str):
                    ip, port = _split_client_addr(client_addr)
                    if port == "0":
                        client_addr = ip
                    if ip:
                        host = get_cached_hostname(ip)
                        new_addr: str | None = None
                        if host and host != ip:
                            new_addr = f"{host} ({client_addr})"
                        elif port == "0":
                            new_addr = client_addr
                        if new_addr is not None:
                            rec = copy.copy(record)
                            rec.args = (new_addr,) + args[1:]
                            return super().formatMessage(rec)
            return super().formatMessage(record)

    return HostnameAccessFormatter


# Built lazily on first attribute access so plain imports of this module
# don't require uvicorn to be importable (e.g. in tests).
_hostname_formatter_cls = None


def _ensure_hostname_formatter():
    global _hostname_formatter_cls
    if _hostname_formatter_cls is None:
        _hostname_formatter_cls = _build_access_formatter_class()
    return _hostname_formatter_cls


def build_logging_config(
    base_config: dict,
    log_level: str,
    log_file: str | None = None,
    reverse_dns_server: str | None = None,
    reverse_dns_sync_wait_ms: int | None = None,
) -> dict:
    """Return a Uvicorn logging config with optional additive file logging."""
    if reverse_dns_server:
        os.environ["_COPILOT_ADAPTER_REVERSE_DNS_SERVER"] = reverse_dns_server
    if reverse_dns_sync_wait_ms is not None:
        os.environ["_COPILOT_ADAPTER_REVERSE_DNS_SYNC_WAIT_MS"] = str(reverse_dns_sync_wait_ms)

    _ensure_hostname_formatter()

    config = copy.deepcopy(base_config)
    config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelprefix)s %(message)s"
    access_formatter = config["formatters"]["access"]
    access_formatter["fmt"] = (
        '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    access_formatter["()"] = "lib.logging.HostnameAccessFormatter"

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
    return os.environ.get("_COPILOT_ADAPTER_LOG_FILE", default)


def build_runtime_logging_config(
    base_config: dict,
    log_level: str,
    log_file: str | None = None,
    reverse_dns_server: str | None = None,
    reverse_dns_sync_wait_ms: int | None = None,
) -> dict:
    """Build logging config, honoring worker env overrides when present."""
    if reverse_dns_sync_wait_ms is None:
        reverse_dns_sync_wait_ms = _reverse_dns_sync_wait_ms() or None
    return build_logging_config(
        base_config,
        log_level,
        resolve_worker_log_file(log_file),
        reverse_dns_server or _reverse_dns_server(),
        reverse_dns_sync_wait_ms,
    )


# Make ``lib.logging.HostnameAccessFormatter`` importable as a class path
# used by the dictConfig ``()`` factory. We lazily build it on first access
# by resolving the module attribute.
def __getattr__(name: str):
    if name == "HostnameAccessFormatter":
        return _ensure_hostname_formatter()
    raise AttributeError(name)


__all__ = [
    "LifespanFilter",
    "build_logging_config",
    "build_runtime_logging_config",
    "resolve_worker_log_file",
    "get_cached_hostname",
]
