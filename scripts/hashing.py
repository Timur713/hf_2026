"""Streaming file hashes compatible with the shipped Python 3.10 image."""

import hashlib


def file_digest(stream, algorithm):
    """Hash a freshly opened binary stream without a Python 3.11-only API."""
    digest = hashlib.new(algorithm)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest
