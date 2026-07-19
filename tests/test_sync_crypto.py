"""Tests for brain.sync.crypto — Fernet payload encryption, passphrase KDF,
and HMAC body integrity.

The crypto-dependent cases are skipped (``pytest.importorskip``) when the
optional ``[sync]`` extra (``cryptography``) is not installed, so the suite
stays green whether or not the dependency is present. Two invariants are
ALWAYS asserted regardless of the dep: (1) ``import brain.sync.crypto``
succeeds on the floor tier, and (2) ``crypto_available()`` returns a bool
without raising.
"""

from __future__ import annotations

import os

import pytest

# Floor-tier invariant: importing the module must NOT require cryptography.
from brain.sync import crypto
from brain.sync.crypto import SyncCrypto, crypto_available, new_salt

# ---------------------------------------------------------------------------
# Always-on tests (no cryptography required)
# ---------------------------------------------------------------------------

def test_module_imports_without_cryptography():
    """The module object exists and exposes the public API at import time."""
    assert hasattr(crypto, "SyncCrypto")
    assert hasattr(crypto, "from_passphrase") or hasattr(crypto.SyncCrypto, "from_passphrase")
    assert callable(crypto.new_salt)
    assert callable(crypto.crypto_available)


def test_crypto_available_returns_bool_without_raising():
    result = crypto_available()
    assert isinstance(result, bool)


def test_new_salt_length_and_randomness():
    assert len(new_salt()) == 16
    assert len(new_salt(32)) == 32
    # Overwhelmingly unlikely to collide — asserts it is actually random.
    assert new_salt() != new_salt()
    with pytest.raises(ValueError):
        new_salt(0)


def test_body_mac_is_stdlib_only():
    """body_mac / verify_mac need no cryptography (pure hmac/hashlib)."""
    sc = SyncCrypto(os.urandom(32))
    data = b"the whole push body"
    mac = sc.body_mac(data)
    assert isinstance(mac, str)
    assert sc.verify_mac(data, mac) is True
    # Tampered body fails.
    assert sc.verify_mac(data + b"x", mac) is False
    # Tampered MAC fails.
    assert sc.verify_mac(data, mac[:-1] + ("0" if mac[-1] != "0" else "1")) is False


# ---------------------------------------------------------------------------
# Crypto-dependent tests (skipped when cryptography is absent)
# ---------------------------------------------------------------------------

pytest.importorskip("cryptography")


def test_encrypt_decrypt_roundtrip():
    sc = SyncCrypto(os.urandom(32))
    plaintext = b"hermes multi-device secret payload"
    token = sc.encrypt(plaintext)
    assert token != plaintext
    assert sc.decrypt(token) == plaintext


def test_decrypt_with_wrong_key_raises():
    from cryptography.fernet import InvalidToken

    sc = SyncCrypto(os.urandom(32))
    other = SyncCrypto(os.urandom(32))
    token = sc.encrypt(b"secret")
    with pytest.raises(InvalidToken):
        other.decrypt(token)


def test_tampered_token_raises():
    from cryptography.fernet import InvalidToken

    sc = SyncCrypto(os.urandom(32))
    token = bytearray(sc.encrypt(b"secret"))
    token[-5] ^= 0x01  # flip a bit deep inside the token
    with pytest.raises(InvalidToken):
        sc.decrypt(bytes(token))


def test_from_passphrase_deterministic_same_salt():
    salt = new_salt()
    a = SyncCrypto.from_passphrase("correct horse battery staple", salt)
    b = SyncCrypto.from_passphrase("correct horse battery staple", salt)
    # Same passphrase + salt -> same key -> cross-decrypt works.
    token = a.encrypt(b"payload")
    assert b.decrypt(token) == b"payload"
    assert a.body_mac(b"x") == b.body_mac(b"x")


def test_from_passphrase_different_salt_differs():
    a = SyncCrypto.from_passphrase("pass", new_salt())
    b = SyncCrypto.from_passphrase("pass", new_salt())
    # Different salt -> different key -> cannot decrypt each other's token.
    from cryptography.fernet import InvalidToken

    token = a.encrypt(b"payload")
    with pytest.raises(InvalidToken):
        b.decrypt(token)


def test_from_passphrase_different_passphrase_differs():
    salt = new_salt()
    a = SyncCrypto.from_passphrase("passphrase-one", salt)
    b = SyncCrypto.from_passphrase("passphrase-two", salt)
    assert a.body_mac(b"x") != b.body_mac(b"x")


def test_accepts_base64_fernet_key():
    from cryptography.fernet import Fernet

    key = Fernet.generate_key()  # 44-byte urlsafe base64 Fernet key
    sc = SyncCrypto(key)
    token = sc.encrypt(b"hi")
    assert sc.decrypt(token) == b"hi"


def test_body_mac_roundtrip_with_derived_key():
    sc = SyncCrypto.from_passphrase("pw", new_salt())
    data = b"body-bytes"
    mac = sc.body_mac(data)
    assert sc.verify_mac(data, mac) is True
    assert sc.verify_mac(b"other", mac) is False
