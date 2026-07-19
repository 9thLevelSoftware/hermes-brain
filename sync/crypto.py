"""Client-side sync crypto: Fernet payload encryption + passphrase KDF + body MAC.

Adapted from mnemosyne-oss/mnemosyne (``mnemosyne/core/sync.py``, MIT,
(c) 2026 Abdias J). Only the encryption / key-derivation / MAC logic is
carried over here; the donor's event log, conflict resolution and relay
transport are owned by other modules and are intentionally discarded.

NOTE on the primitive: the donor markets "XChaCha20" but the actual code
uses ``cryptography.fernet.Fernet`` (AES-128-CBC + HMAC-SHA256). This module
faithfully implements FERNET, matching the real donor code rather than the
marketing copy. The relay never holds a key — it only ever sees Fernet
tokens, so all crypto here is strictly client-side.

Floor-tier rule (invariant): this module lives behind the optional ``[sync]``
extra. ``cryptography`` (and ``argon2``) are imported LAZILY inside the
methods that need them, so ``import brain.sync.crypto`` succeeds on the
stdlib floor tier with no optional deps installed. A crypto operation
attempted without the dependency raises a clear ``RuntimeError`` naming the
``[sync]`` extra. Only ``hmac``/``hashlib``/``base64``/``os`` (stdlib) are
imported at module level.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

# Message shown when a crypto operation is attempted without the optional dep.
_SYNC_EXTRA_MSG = (
    "sync crypto requires the optional 'cryptography' dependency; "
    "install it with:  pip install -e .[sync]"
)


def crypto_available() -> bool:
    """Return True iff ``cryptography`` is importable. Never raises."""
    try:
        import cryptography  # noqa: F401  (probe only)

        return True
    except Exception:  # pragma: no cover - defensive; import errors of any shape
        return False


def new_salt(n: int = 16) -> bytes:
    """Return ``n`` cryptographically random bytes for use as a KDF salt."""
    if n <= 0:
        raise ValueError("salt length must be positive")
    return os.urandom(n)


def _coerce_fernet_key(key: bytes) -> bytes:
    """Normalize *key* into a urlsafe-base64 32-byte Fernet key.

    Accepts either an already-encoded Fernet key (44-byte urlsafe base64 that
    decodes to exactly 32 bytes) or a raw 32-byte seed (which is then
    base64-encoded). Returns the base64 Fernet key as ``bytes``.
    """
    if isinstance(key, str):
        key = key.encode("ascii")
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes or str")
    key = bytes(key)

    # Already a Fernet key? (urlsafe base64 that decodes to 32 raw bytes.)
    try:
        decoded = base64.urlsafe_b64decode(key)
        if len(decoded) == 32:
            return key
    except (ValueError, TypeError):
        # binascii.Error (from an invalid base64 seed) is a ValueError subclass.
        pass

    # Otherwise treat it as a raw 32-byte seed.
    if len(key) == 32:
        return base64.urlsafe_b64encode(key)

    raise ValueError(
        "key must be a 32-byte raw seed or a urlsafe-base64 Fernet key"
    )


class SyncCrypto:
    """Fernet encryption + HMAC body integrity for the sync push body.

    Construct from a raw 32-byte seed / base64 Fernet key, or derive from a
    passphrase via :meth:`from_passphrase`.
    """

    def __init__(self, key: bytes) -> None:
        # Store the base64 Fernet key and the raw 32-byte material (the latter
        # keys the HMAC body-MAC). No cryptography import needed here — the key
        # is just bytes until an actual encrypt/decrypt happens.
        self._fernet_key: bytes = _coerce_fernet_key(key)
        self._raw_key: bytes = base64.urlsafe_b64decode(self._fernet_key)

    # -- key derivation ----------------------------------------------------

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes) -> SyncCrypto:
        """Derive the key from a passphrase + salt (deterministic).

        Prefers **Argon2id** when available — first ``cryptography``'s own
        ``Argon2id`` KDF (cryptography >= 44), then the ``argon2-cffi``
        low-level binding. Falls back to **PBKDF2-HMAC-SHA256** at 600,000
        iterations when neither Argon2 backend is present. All three derive a
        32-byte key; the same passphrase+salt always yields the same key.

        Argon2id params: time_cost=3, memory_cost=65536 KiB (64 MiB),
        parallelism=4, hash_len=32 (OWASP-recommended baseline).
        """
        if not isinstance(passphrase, str):
            raise TypeError("passphrase must be str")
        if not isinstance(salt, (bytes, bytearray)) or len(salt) == 0:
            raise ValueError("salt must be non-empty bytes")
        salt = bytes(salt)
        secret = passphrase.encode("utf-8")

        # 1) cryptography's Argon2id (preferred).
        try:
            from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

            kdf = Argon2id(
                salt=salt,
                length=32,
                iterations=3,
                lanes=4,
                memory_cost=65536,
            )
            return cls(kdf.derive(secret))
        except ImportError:
            pass

        # 2) argon2-cffi low-level (Argon2id) — same primitive, different lib.
        try:
            import argon2.low_level as _argon2

            raw = _argon2.hash_secret_raw(
                secret=secret,
                salt=salt,
                time_cost=3,
                memory_cost=65536,
                parallelism=4,
                hash_len=32,
                type=_argon2.Type.ID,
            )
            return cls(raw)
        except ImportError:
            pass

        # 3) PBKDF2-HMAC-SHA256 fallback (needs cryptography).
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError as exc:
            raise RuntimeError(_SYNC_EXTRA_MSG) from exc

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600_000,
        )
        return cls(kdf.derive(secret))

    # -- payload encryption ------------------------------------------------

    def _fernet(self):
        """Return a Fernet instance, or raise a clear error without the dep."""
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError(_SYNC_EXTRA_MSG) from exc
        return Fernet(self._fernet_key)

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt *plaintext* into a Fernet token (bytes)."""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        return self._fernet().encrypt(plaintext)

    def decrypt(self, token: bytes) -> bytes:
        """Decrypt a Fernet *token*. Raises on tamper / wrong key.

        ``cryptography.fernet.InvalidToken`` propagates for a bad key or a
        modified token — callers treat that as an authentication failure.
        """
        if isinstance(token, str):
            token = token.encode("utf-8")
        return self._fernet().decrypt(token)

    # -- body integrity (whole push body) ----------------------------------

    def body_mac(self, data: bytes) -> str:
        """Return the HMAC-SHA256 hex digest of *data* keyed by the sync key.

        Stdlib-only (``hmac``/``hashlib``); needs no ``cryptography``. Used to
        authenticate the whole push body around the individually encrypted
        payloads.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hmac.new(self._raw_key, data, hashlib.sha256).hexdigest()

    def verify_mac(self, data: bytes, mac: str) -> bool:
        """Constant-time check that *mac* is the body MAC of *data*."""
        if isinstance(mac, bytes):
            mac = mac.decode("ascii", "replace")
        expected = self.body_mac(data)
        return hmac.compare_digest(expected, mac)
