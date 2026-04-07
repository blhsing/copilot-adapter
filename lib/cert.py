"""Self-signed CA and per-host server certificate generation for MITM proxy."""

import ssl
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

DEFAULT_CA_DIR = Path.home() / ".config" / "copilot-api"

_cert_cache: dict[str, tuple[x509.Certificate, rsa.RSAPrivateKey]] = {}
_cache_lock = threading.Lock()


def ca_paths(ca_dir: Path | None = None) -> tuple[Path, Path]:
    """Return (cert_path, key_path) for the CA."""
    d = ca_dir or DEFAULT_CA_DIR
    return d / "ca.pem", d / "ca-key.pem"


def ensure_ca(ca_dir: Path | None = None) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    """Load or generate the MITM CA certificate and key."""
    cert_path, key_path = ca_paths(ca_dir)

    if cert_path.exists() and key_path.exists():
        ca_cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        ca_key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        return ca_cert, ca_key

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "copilot-adapter MITM CA")])
    now = datetime.now(timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    d = ca_dir or DEFAULT_CA_DIR
    d.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM,
                          serialization.PrivateFormat.TraditionalOpenSSL,
                          serialization.NoEncryption())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return cert, key


def generate_server_cert(
    hostname: str, ca_cert: x509.Certificate, ca_key: rsa.RSAPrivateKey,
) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    """Generate (or return cached) a server certificate for *hostname* signed by the CA."""
    with _cache_lock:
        if hostname in _cert_cache:
            return _cert_cache[hostname]

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    with _cache_lock:
        _cert_cache[hostname] = (cert, key)
    return cert, key


def build_server_ssl_context(
    hostname: str, ca_cert: x509.Certificate, ca_key: rsa.RSAPrivateKey,
) -> ssl.SSLContext:
    """Return an SSL context for serving TLS as *hostname*."""
    cert, key = generate_server_cert(hostname, ca_cert, ca_key)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Load from in-memory PEM via tempfiles (cross-platform)
    import tempfile, os
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    # Write to temp files, load, then delete
    cert_fd, cert_path = tempfile.mkstemp(suffix=".pem")
    key_fd, key_path = tempfile.mkstemp(suffix=".pem")
    try:
        os.write(cert_fd, cert_pem)
        os.close(cert_fd)
        os.write(key_fd, key_pem)
        os.close(key_fd)
        ctx.load_cert_chain(cert_path, key_path)
    finally:
        os.unlink(cert_path)
        os.unlink(key_path)

    return ctx
