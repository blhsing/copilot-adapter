"""Dual-mode server: forward HTTP/HTTPS proxy + reverse API proxy on one port.

When ``--proxy`` is enabled the TCP listener inspects each connection's first
request line:

* ``CONNECT host:port`` → forward-proxy path (MITM for Copilot, blind relay
  for everything else).
* Absolute URL (``GET http://…``) → plain HTTP forward proxy.
* Relative path (``POST /v1/chat/completions``) → handed to the FastAPI /
  Uvicorn ASGI app (the normal reverse-proxy behaviour).
"""

import asyncio
import logging
import os
import ssl
import socket
from pathlib import Path
from urllib.parse import urlparse

import httpx
import uvicorn

from .cert import ca_paths, ensure_ca, build_server_ssl_context

logger = logging.getLogger(__name__)

COPILOT_HOST = "api.githubcopilot.com"
_INTERCEPT_HOSTS = {
    COPILOT_HOST,
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
}
_BUF = 65536


def _get_upstream_proxy() -> tuple[str, int] | None:
    """Read HTTPS_PROXY / HTTP_PROXY from environment and return (host, port)."""
    raw = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") \
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not raw:
        return None
    parsed = urlparse(raw)
    host = parsed.hostname
    port = parsed.port or 80
    if not host:
        return None
    return host, port


async def _open_upstream(host: str, port: int, *,
                         tls: bool = False,
                         ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to host:port, tunnelling through the upstream proxy if set.

    For TLS connections through a proxy, we send CONNECT, wait for 200,
    then wrap with SSL.
    """
    upstream = _get_upstream_proxy()

    if upstream is None:
        # Direct connection
        ssl_ctx = ssl.create_default_context() if tls else None
        return await asyncio.open_connection(host, port, ssl=ssl_ctx)

    proxy_host, proxy_port = upstream
    reader, writer = await asyncio.open_connection(proxy_host, proxy_port)

    # Ask the upstream proxy to tunnel
    writer.write(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    await writer.drain()

    # Read the proxy response
    resp_line = await reader.readline()
    if not resp_line:
        writer.close()
        raise ConnectionError(f"Upstream proxy closed connection for CONNECT {host}:{port}")

    # Consume response headers
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break

    # Check for 200
    if b"200" not in resp_line:
        writer.close()
        raise ConnectionError(
            f"Upstream proxy rejected CONNECT {host}:{port}: {resp_line.decode(errors='replace').strip()}"
        )

    if tls:
        # TLS upgrade over the tunnel using start_tls
        ssl_ctx = ssl.create_default_context()
        loop = asyncio.get_event_loop()
        transport = writer.transport
        protocol = transport.get_protocol()
        new_transport = await loop.start_tls(
            transport, protocol, ssl_ctx, server_hostname=host,
        )
        # Create a fresh reader/protocol — start_tls breaks the old reader
        new_reader = asyncio.StreamReader()
        new_protocol = asyncio.StreamReaderProtocol(new_reader)
        new_protocol.connection_made(new_transport)
        new_transport.set_protocol(new_protocol)
        new_writer = asyncio.StreamWriter(new_transport, new_protocol, new_reader, loop)
        return new_reader, new_writer

    return reader, writer


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

async def _read_request_line(reader: asyncio.StreamReader) -> bytes | None:
    """Read the first HTTP request line (e.g. ``CONNECT host:443 HTTP/1.1``)."""
    try:
        line = await asyncio.wait_for(reader.readline(), timeout=30)
    except (asyncio.TimeoutError, ConnectionError):
        return None
    return line if line else None


async def _read_headers(reader: asyncio.StreamReader) -> list[tuple[bytes, bytes]]:
    """Read HTTP headers until the blank line.  Return list of (name, value)."""
    headers: list[tuple[bytes, bytes]] = []
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        if b":" in line:
            name, _, value = line.partition(b":")
            headers.append((name.strip(), value.strip()))
    return headers


async def _read_body(reader: asyncio.StreamReader,
                     headers: list[tuple[bytes, bytes]]) -> bytes:
    """Read the request body based on Content-Length."""
    for name, value in headers:
        if name.lower() == b"content-length":
            length = int(value)
            return await reader.readexactly(length)
    return b""


def _serialize_request(request_line: bytes,
                       headers: list[tuple[bytes, bytes]],
                       body: bytes) -> bytes:
    """Reassemble an HTTP request from parsed components."""
    parts = [request_line.rstrip(b"\r\n")]
    for name, value in headers:
        parts.append(name + b": " + value)
    parts.append(b"")
    parts.append(b"")
    result = b"\r\n".join(parts)
    return result + body


def _rewrite_initiator(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """Rewrite X-Initiator: user → agent."""
    out = []
    for name, value in headers:
        if name.lower() == b"x-initiator" and value.lower() == b"user":
            logger.debug("Rewriting X-Initiator: user -> agent")
            out.append((name, b"agent"))
        else:
            out.append((name, value))
    return out


# ---------------------------------------------------------------------------
# Blind relay (non-Copilot CONNECT targets)
# ---------------------------------------------------------------------------

async def _relay(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
    try:
        while True:
            data = await src.read(_BUF)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            dst.close()
        except Exception:
            pass


async def _blind_relay(r1: asyncio.StreamReader, w1: asyncio.StreamWriter,
                       r2: asyncio.StreamReader, w2: asyncio.StreamWriter):
    await asyncio.gather(_relay(r1, w2), _relay(r2, w1),
                         return_exceptions=True)


# ---------------------------------------------------------------------------
# MITM path for Copilot
# ---------------------------------------------------------------------------

async def _tls_upgrade_server(reader: asyncio.StreamReader,
                              writer: asyncio.StreamWriter,
                              ssl_ctx: ssl.SSLContext,
                              ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    """Upgrade an existing connection to TLS (server-side) using start_tls."""
    loop = asyncio.get_event_loop()
    transport = writer.transport
    protocol = transport.get_protocol()

    try:
        new_transport = await loop.start_tls(
            transport, protocol, ssl_ctx, server_side=True,
        )
    except (ssl.SSLError, OSError) as exc:
        logger.debug("TLS handshake failed: %s", exc)
        writer.close()
        return None

    new_writer = asyncio.StreamWriter(new_transport, protocol, reader, loop)
    return reader, new_writer


async def _handle_copilot_mitm(client_reader: asyncio.StreamReader,
                               client_writer: asyncio.StreamWriter):
    """Parse HTTP requests from the decrypted client stream, rewrite
    X-Initiator, and relay to the real Copilot API."""

    # Open a real TLS connection to the Copilot API (via upstream proxy if set)
    try:
        up_reader, up_writer = await _open_upstream(
            COPILOT_HOST, 443, tls=True,
        )
    except Exception as exc:
        logger.error("Failed to connect to %s: %s", COPILOT_HOST, exc)
        client_writer.close()
        return

    try:
        while True:
            # Read one HTTP request from the client
            request_line = await _read_request_line(client_reader)
            if not request_line:
                break

            headers = await _read_headers(client_reader)
            body = await _read_body(client_reader, headers)

            # Rewrite the initiator header
            headers = _rewrite_initiator(headers)

            # Forward to upstream
            raw = _serialize_request(request_line, headers, body)
            up_writer.write(raw)
            await up_writer.drain()

            # Relay the response — read the response header block, then body
            resp_line = await _read_request_line(up_reader)
            if not resp_line:
                break
            resp_headers = await _read_headers(up_reader)

            # Determine response body framing
            client_writer.write(resp_line)
            for name, value in resp_headers:
                client_writer.write(name + b": " + value + b"\r\n")
            client_writer.write(b"\r\n")
            await client_writer.drain()

            # Check for chunked transfer encoding
            is_chunked = any(
                name.lower() == b"transfer-encoding" and b"chunked" in value.lower()
                for name, value in resp_headers
            )
            content_length = None
            for name, value in resp_headers:
                if name.lower() == b"content-length":
                    content_length = int(value)
                    break

            if is_chunked:
                # Relay chunked response
                while True:
                    chunk_header = await up_reader.readline()
                    if not chunk_header:
                        break
                    client_writer.write(chunk_header)
                    await client_writer.drain()

                    # Parse chunk size
                    try:
                        chunk_size = int(chunk_header.strip(), 16)
                    except ValueError:
                        continue
                    if chunk_size == 0:
                        # Read trailing \r\n
                        trailer = await up_reader.readline()
                        client_writer.write(trailer)
                        await client_writer.drain()
                        break
                    # Read chunk data + trailing \r\n
                    chunk_data = await up_reader.readexactly(chunk_size + 2)
                    client_writer.write(chunk_data)
                    await client_writer.drain()
            elif content_length is not None:
                # Fixed-length body
                remaining = content_length
                while remaining > 0:
                    data = await up_reader.read(min(remaining, _BUF))
                    if not data:
                        break
                    client_writer.write(data)
                    await client_writer.drain()
                    remaining -= len(data)
            else:
                # No content-length and not chunked — read until connection
                # closes (uncommon for keep-alive, but handle it)
                # For keep-alive we assume the response is empty if neither
                # content-length nor chunked is specified
                pass

            # Check connection keep-alive
            connection = b"keep-alive"
            for name, value in resp_headers:
                if name.lower() == b"connection":
                    connection = value.lower()
                    break
            if connection == b"close":
                break

    except (ConnectionError, OSError, asyncio.IncompleteReadError):
        pass
    finally:
        for w in (client_writer, up_writer):
            try:
                w.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MITM path for OpenAI / Anthropic / Gemini → internal Uvicorn or upstream
# ---------------------------------------------------------------------------

# Paths that should be routed to the internal adapter
_API_PATH_PREFIXES = (
    # OpenAI
    b"/v1/chat/completions",
    b"/v1/responses",
    b"/v1/embeddings",
    b"/v1/models",
    # Anthropic
    b"/v1/messages",
    # Gemini
    b"/v1beta/models",
    # Legacy / catch-all
    b"/chat/completions",
    b"/responses",
    b"/messages",
    b"/embeddings",
    b"/models",
)


def _is_api_path(request_line: bytes) -> bool:
    """Check if the request path matches a known LLM API endpoint."""
    parts = request_line.split()
    if len(parts) < 2:
        return False
    path = parts[1].split(b"?")[0]
    return any(path == p or path.startswith(p + b"/")
               for p in _API_PATH_PREFIXES)


async def _handle_rewrite_mitm(client_reader: asyncio.StreamReader,
                               client_writer: asyncio.StreamWriter,
                               original_host: str,
                               internal_host: str,
                               internal_port: int):
    """Parse HTTP requests from a MITM'd API connection and forward them
    to the internal Uvicorn server (for API paths) or the original host
    (for everything else)."""
    try:
        while True:
            request_line = await _read_request_line(client_reader)
            if not request_line:
                break

            headers = await _read_headers(client_reader)
            body = await _read_body(client_reader, headers)

            if _is_api_path(request_line):
                # Route to internal Uvicorn (reroute to Copilot)
                req_desc = request_line.rstrip(b"\r\n").decode("ascii", errors="replace")
                logger.debug("Rerouting %s -> %s to adapter", original_host, req_desc)
                fwd_headers = [
                    (n, v) for n, v in headers
                    if n.lower() != b"host"
                ]
                host_val = f"{internal_host}:{internal_port}".encode()
                fwd_headers.append((b"Host", host_val))
                raw = _serialize_request(request_line, fwd_headers, body)
                try:
                    up_reader, up_writer = await asyncio.open_connection(
                        internal_host, internal_port,
                    )
                except Exception as exc:
                    logger.error("Failed to connect to internal Uvicorn: %s", exc)
                    client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                    await client_writer.drain()
                    break
            else:
                # Route to the original upstream host via httpx (passthrough)
                req_desc = request_line.rstrip(b"\r\n").decode("ascii", errors="replace")
                logger.debug("Passing through %s -> %s to %s", original_host, req_desc, original_host)

                parts = request_line.split()
                method_str = parts[0].decode("ascii")
                path_str = parts[1].decode("ascii", errors="replace")
                url = f"https://{original_host}{path_str}"

                # Convert headers to dict (httpx format)
                hdr_dict = {}
                for n, v in headers:
                    name_str = n.decode("ascii", errors="replace")
                    # Skip hop-by-hop headers
                    if name_str.lower() in ("transfer-encoding", "connection",
                                            "keep-alive", "proxy-connection"):
                        continue
                    hdr_dict[name_str] = v.decode("ascii", errors="replace")

                try:
                    async with httpx.AsyncClient(proxy=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")) as hx:
                        resp = await hx.request(
                            method_str, url, headers=hdr_dict,
                            content=body if body else None,
                            follow_redirects=False,
                        )
                except Exception as exc:
                    logger.error("Passthrough to %s failed: %s", original_host, exc)
                    client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                    await client_writer.drain()
                    break

                # Write HTTP response back to client
                status_line = f"HTTP/1.1 {resp.status_code} {resp.reason_phrase}\r\n".encode()
                client_writer.write(status_line)
                resp_body = resp.content
                for key, val in resp.headers.raw:
                    # Skip headers we rewrite or that are hop-by-hop;
                    # also skip content-encoding/content-length since httpx
                    # auto-decompresses and we set our own content-length
                    if key.lower() in (b"transfer-encoding", b"connection",
                                       b"content-encoding", b"content-length"):
                        continue
                    client_writer.write(key + b": " + val + b"\r\n")
                client_writer.write(f"Content-Length: {len(resp_body)}\r\n".encode())
                client_writer.write(b"Connection: keep-alive\r\n")
                client_writer.write(b"\r\n")
                client_writer.write(resp_body)
                await client_writer.drain()

                logger.debug("Passthrough response: %d %s", resp.status_code, resp.reason_phrase)
                continue

            up_writer.write(raw)
            await up_writer.drain()

            # Relay response back to client
            resp_line = await _read_request_line(up_reader)
            if not resp_line:
                up_writer.close()
                break

            resp_headers = await _read_headers(up_reader)

            client_writer.write(resp_line)
            for name, value in resp_headers:
                client_writer.write(name + b": " + value + b"\r\n")
            client_writer.write(b"\r\n")
            await client_writer.drain()

            # Determine response body framing
            is_chunked = any(
                name.lower() == b"transfer-encoding" and b"chunked" in value.lower()
                for name, value in resp_headers
            )
            content_length = None
            for name, value in resp_headers:
                if name.lower() == b"content-length":
                    content_length = int(value)
                    break

            if is_chunked:
                while True:
                    chunk_header = await up_reader.readline()
                    if not chunk_header:
                        break
                    client_writer.write(chunk_header)
                    await client_writer.drain()
                    try:
                        chunk_size = int(chunk_header.strip(), 16)
                    except ValueError:
                        continue
                    if chunk_size == 0:
                        trailer = await up_reader.readline()
                        client_writer.write(trailer)
                        await client_writer.drain()
                        break
                    chunk_data = await up_reader.readexactly(chunk_size + 2)
                    client_writer.write(chunk_data)
                    await client_writer.drain()
            elif content_length is not None:
                remaining = content_length
                while remaining > 0:
                    data = await up_reader.read(min(remaining, _BUF))
                    if not data:
                        break
                    client_writer.write(data)
                    await client_writer.drain()
                    remaining -= len(data)

            up_writer.close()

            # Check connection keep-alive
            connection = b"keep-alive"
            for name, value in resp_headers:
                if name.lower() == b"connection":
                    connection = value.lower()
                    break
            if connection == b"close":
                break

    except (ConnectionError, OSError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            client_writer.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Connection handler (dispatches CONNECT / plain-HTTP / ASGI)
# ---------------------------------------------------------------------------

async def _handle_connect(host: str, port: int,
                          reader: asyncio.StreamReader,
                          writer: asyncio.StreamWriter,
                          ca_cert, ca_key,
                          internal_host: str = "127.0.0.1",
                          internal_port: int = 0):
    """Handle a CONNECT tunnel request."""
    # Consume remaining headers
    await _read_headers(reader)
    await _do_connect(host, port, reader, writer, ca_cert, ca_key,
                      internal_host, internal_port)


async def _handle_connect_pre_read(host: str, port: int,
                                   reader: asyncio.StreamReader,
                                   writer: asyncio.StreamWriter,
                                   ca_cert, ca_key,
                                   internal_host: str = "127.0.0.1",
                                   internal_port: int = 0):
    """Handle a CONNECT request when headers have already been consumed."""
    await _do_connect(host, port, reader, writer, ca_cert, ca_key,
                      internal_host, internal_port)


async def _do_connect(host: str, port: int,
                      reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter,
                      ca_cert, ca_key,
                      internal_host: str, internal_port: int):
    """Core CONNECT handling after headers are consumed."""
    # Respond 200
    writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await writer.drain()

    if host in _INTERCEPT_HOSTS:
        logger.info("MITM intercepting CONNECT to %s:%d", host, port)
        ssl_ctx = build_server_ssl_context(host, ca_cert, ca_key)
        result = await _tls_upgrade_server(reader, writer, ssl_ctx)
        if result is None:
            return
        tls_reader, tls_writer = result
        if host == COPILOT_HOST:
            await _handle_copilot_mitm(tls_reader, tls_writer)
        else:
            await _handle_rewrite_mitm(tls_reader, tls_writer,
                                       host, internal_host, internal_port)
    else:
        logger.debug("Blind relay CONNECT to %s:%d", host, port)
        try:
            up_reader, up_writer = await _open_upstream(host, port)
        except Exception as exc:
            logger.error("Failed to connect to %s:%d: %s", host, port, exc)
            writer.close()
            return
        await _blind_relay(reader, writer, up_reader, up_writer)


async def _handle_plain_http(method: bytes, url: bytes, version: bytes,
                             reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter):
    """Handle a plain HTTP request with an absolute URL (forward proxy)."""
    headers = await _read_headers(reader)
    await _do_plain_http(method, url, version, headers, reader, writer)


async def _handle_plain_http_pre_read(method: bytes, url: bytes, version: bytes,
                                      headers: list[tuple[bytes, bytes]],
                                      reader: asyncio.StreamReader,
                                      writer: asyncio.StreamWriter):
    """Handle a plain HTTP request when headers have already been consumed."""
    await _do_plain_http(method, url, version, headers, reader, writer)


async def _do_plain_http(method: bytes, url: bytes, version: bytes,
                         headers: list[tuple[bytes, bytes]],
                         reader: asyncio.StreamReader,
                         writer: asyncio.StreamWriter):
    """Core plain HTTP forward proxy handling."""
    parsed = urlparse(url)
    host = parsed.hostname or b""
    port = parsed.port or 80
    path = parsed.path or b"/"
    if parsed.query:
        path = path + b"?" + parsed.query

    if isinstance(host, bytes):
        host_str = host.decode("ascii", errors="replace")
    else:
        host_str = host

    body = await _read_body(reader, headers)

    # Rewrite request line to use relative path
    request_line = method + b" " + path + b" " + version + b"\r\n"

    # Rewrite initiator if targeting Copilot
    if host_str == COPILOT_HOST:
        headers = _rewrite_initiator(headers)

    raw = _serialize_request(request_line, headers, body)

    try:
        up_reader, up_writer = await _open_upstream(host_str, port)
    except Exception as exc:
        logger.error("Failed to connect to %s:%d: %s", host_str, port, exc)
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        writer.close()
        return

    up_writer.write(raw)
    await up_writer.drain()

    # Relay the entire response back
    await _relay(up_reader, writer)
    up_writer.close()


# ---------------------------------------------------------------------------
# Dual-mode server
# ---------------------------------------------------------------------------

class DualModeServer:
    """TCP server that dispatches connections to forward-proxy or ASGI."""

    def __init__(self, asgi_app, *, host: str, port: int,
                 ca_dir: Path | None = None,
                 proxy_user: str | None = None,
                 proxy_password: str | None = None,
                 uvicorn_log_level: str = "info",
                 uvicorn_log_config: dict | None = None,
                 uvicorn_use_colors: bool = True,
                 timeout_graceful_shutdown: int = 5):
        self._asgi_app = asgi_app
        self._host = host
        self._port = port
        self._ca_dir = ca_dir
        self._uvicorn_log_level = uvicorn_log_level
        self._uvicorn_log_config = uvicorn_log_config
        self._uvicorn_use_colors = uvicorn_use_colors
        self._timeout = timeout_graceful_shutdown

        # Proxy auth
        self._proxy_auth: str | None = None
        if proxy_user and proxy_password:
            import base64
            cred = base64.b64encode(
                f"{proxy_user}:{proxy_password}".encode()
            ).decode()
            self._proxy_auth = cred

        # CA for MITM
        self._ca_cert, self._ca_key = ensure_ca(ca_dir)

        # Internal Uvicorn server on localhost ephemeral port
        self._internal_host = "127.0.0.1"
        self._internal_port: int | None = None
        self._uvicorn_server: uvicorn.Server | None = None

    async def _start_uvicorn(self):
        """Start Uvicorn on an ephemeral port and record the actual port."""
        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self._internal_host, 0))
            self._internal_port = s.getsockname()[1]

        config = uvicorn.Config(
            self._asgi_app,
            host=self._internal_host,
            port=self._internal_port,
            log_level=self._uvicorn_log_level,
            log_config=self._uvicorn_log_config,
            timeout_graceful_shutdown=self._timeout,
            use_colors=self._uvicorn_use_colors,
        )
        self._uvicorn_server = uvicorn.Server(config)
        asyncio.create_task(self._uvicorn_server.serve())
        # Wait for Uvicorn to be ready
        while not self._uvicorn_server.started:
            await asyncio.sleep(0.05)

    async def _forward_to_uvicorn(self, first_line: bytes,
                                  reader: asyncio.StreamReader,
                                  writer: asyncio.StreamWriter):
        """Forward a normal API request to the internal Uvicorn server."""
        try:
            up_reader, up_writer = await asyncio.open_connection(
                self._internal_host, self._internal_port,
            )
        except Exception as exc:
            logger.error("Failed to connect to internal Uvicorn: %s", exc)
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            writer.close()
            return

        # Send the first line we already consumed
        up_writer.write(first_line)
        await up_writer.drain()

        # Bidirectional relay for the rest
        await _blind_relay(reader, writer, up_reader, up_writer)

    async def _handle_client(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter):
        """Dispatch a new connection based on the first request line."""
        first_line = await _read_request_line(reader)
        if not first_line:
            writer.close()
            return

        parts = first_line.split()
        if len(parts) < 2:
            writer.close()
            return

        method = parts[0].upper()
        is_forward_proxy = (
            method == b"CONNECT"
            or parts[1].startswith(b"http://")
            or parts[1].startswith(b"https://")
        )

        # Check proxy auth for forward proxy requests
        if is_forward_proxy and self._proxy_auth is not None:
            # Read headers to find Proxy-Authorization
            headers = await _read_headers(reader)
            authed = False
            for name, value in headers:
                if name.lower() == b"proxy-authorization":
                    # Expected: Basic <base64>
                    auth_parts = value.split(None, 1)
                    if (len(auth_parts) == 2
                            and auth_parts[0].lower() == b"basic"
                            and auth_parts[1].decode("ascii", errors="replace") == self._proxy_auth):
                        authed = True
                    break
            if not authed:
                writer.write(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                    b"Proxy-Authenticate: Basic realm=\"copilot-adapter\"\r\n"
                    b"\r\n"
                )
                await writer.drain()
                writer.close()
                return

            # Headers already consumed; dispatch with them
            if method == b"CONNECT":
                target = parts[1].decode("ascii", errors="replace")
                if ":" in target:
                    host, port_str = target.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = target
                    port = 443
                # _handle_connect normally reads headers itself; pass pre-read
                await _handle_connect_pre_read(host, port, reader, writer,
                                               self._ca_cert, self._ca_key,
                                               self._internal_host, self._internal_port)
            else:
                await _handle_plain_http_pre_read(
                    method, parts[1],
                    parts[2] if len(parts) > 2 else b"HTTP/1.1",
                    headers, reader, writer)
            return

        if method == b"CONNECT":
            # CONNECT host:port HTTP/1.1
            target = parts[1].decode("ascii", errors="replace")
            if ":" in target:
                host, port_str = target.rsplit(":", 1)
                port = int(port_str)
            else:
                host = target
                port = 443
            await _handle_connect(host, port, reader, writer,
                                  self._ca_cert, self._ca_key,
                                  self._internal_host, self._internal_port)

        elif parts[1].startswith(b"http://") or parts[1].startswith(b"https://"):
            # Absolute URL → plain HTTP forward proxy
            await _handle_plain_http(method, parts[1],
                                     parts[2] if len(parts) > 2 else b"HTTP/1.1",
                                     reader, writer)
        else:
            # Relative path → normal API proxy, forward to Uvicorn
            await self._forward_to_uvicorn(first_line, reader, writer)

    async def serve(self):
        """Start both the TCP front-end and the internal Uvicorn server."""
        await self._start_uvicorn()

        server = await asyncio.start_server(
            self._handle_client, self._host, self._port,
        )

        cert_path, _ = ca_paths(self._ca_dir)
        logger.info("Forward proxy active on %s:%d", self._host, self._port)
        print(f"\nForward proxy active (same port)")
        print(f"  Intercepts: {', '.join(sorted(_INTERCEPT_HOSTS))}")
        print(f"  CA certificate: {cert_path}")
        print(f"  Configure client: HTTPS_PROXY=http://{self._host}:{self._port}")
        print(f"  Trust CA (Node.js): NODE_EXTRA_CA_CERTS={cert_path}\n")

        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            server.close()
            await server.wait_closed()
            if self._uvicorn_server:
                self._uvicorn_server.should_exit = True
