# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture and validate immutable dataset metadata before distributed construction."""

from __future__ import annotations

import base64
import contextlib
import contextvars
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

_MAX_BYTES = 64 * 1024 * 1024


def _metadata_path(path: str) -> bool:
    return Path(path).suffix in (".yaml", ".yml", ".json") or path.endswith("/.nv-meta/index.uuid")


def _storage_root() -> str | None:
    # EPath imports this module; resolve its effective DSS setting after import.
    from megatron.energon.epathlib.epath import NVDATASET_CACHE_DIR

    return None if NVDATASET_CACHE_DIR is None else str(NVDATASET_CACHE_DIR)


@dataclass(frozen=True)
class MetadataSnapshot:
    """Metadata for one immutable dataset revision; validate before replaying."""

    recipe_identity: str
    contents: dict[str, bytes]
    file_checks: dict[str, bool]
    sizes: dict[str, int]
    jsonl_sizes: dict[str, int]
    storage_root: str | None
    sha256: str

    @classmethod
    def load(cls, path: str | Path, *, sha256: str, recipe_identity: str) -> MetadataSnapshot:
        """Check the selected package, without claiming its sources are current."""
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("Metadata snapshot SHA256 mismatch")
        document = json.loads(payload)
        if document["format_version"] != 2:
            raise ValueError("Unsupported metadata snapshot format; recapture with this version")
        if document["recipe_identity"] != recipe_identity:
            raise ValueError("Metadata snapshot recipe identity mismatch")
        return cls(
            recipe_identity=recipe_identity,
            contents={
                key: base64.b64decode(value, validate=True)
                for key, value in document["contents"].items()
            },
            file_checks=document["file_checks"],
            sizes=document["sizes"],
            jsonl_sizes=document["jsonl_sizes"],
            storage_root=document["storage_root"],
            sha256=sha256,
        )

    def check_storage_root(self) -> None:
        """Reject replay when DSS URLs resolve against a different root."""
        if self.storage_root != _storage_root():
            raise ValueError("Metadata snapshot NVDATASET_CACHE_DIR mismatch")

    def validate(self) -> ValidatedMetadataSnapshot:
        """Read live construction metadata once before distributing this snapshot.

        Detects recipe/metadata edits, missing files, changed sizes, JSONL terminal
        offsets and regenerated WebDataset indexes. This is not a content hash of
        sample payloads: data and indexes must remain immutable throughout the job.
        """
        from megatron.energon.epathlib import EPath
        from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader

        if active_snapshot() is not None or active_capture() is not None:
            raise RuntimeError("Validate outside metadata capture/replay contexts")
        self.check_storage_root()
        for key, expected in self.file_checks.items():
            if EPath(key).is_file() != expected:
                raise ValueError(f"Stale metadata snapshot: file existence changed: {key}")
        for key, expected_size in self.sizes.items():
            if EPath(key).size() != expected_size:
                raise ValueError(f"Stale metadata snapshot: file size changed: {key}")
        for key, expected_content in self.contents.items():
            with EPath(key).open("rb") as stream:
                actual = stream.read(len(expected_content) + 1)
            if actual != expected_content:
                raise ValueError(f"Stale metadata snapshot: contents changed: {key}")
        for key, expected_offset in self.jsonl_sizes.items():
            path = EPath(key)
            if path.size() != expected_offset or IJsonlIndexReader.size(path) != expected_offset:
                raise ValueError(f"Stale metadata snapshot: JSONL index changed: {key}")
        return ValidatedMetadataSnapshot(self)


@dataclass(frozen=True)
class ValidatedMetadataSnapshot:
    """Result of job preflight; distribute to trusted workers, never checkpoint it."""

    snapshot: MetadataSnapshot


@dataclass
class MetadataReplay:
    """One construction scope, including metadata coverage counters."""

    snapshot: MetadataSnapshot
    hits: int = 0
    misses: int = 0
    passthrough_opens: int = 0

    def lookup(self, kind: str, path: str) -> bool | int:
        entries = getattr(self.snapshot, kind)
        if path not in entries:
            self.misses += 1
            raise ValueError(f"Metadata snapshot missing {kind}: {path}; recapture all splits")
        self.hits += 1
        return entries[path]

    def open(self, path: str, mode: str):
        if mode not in ("r", "rb") or not _metadata_path(path):
            self.passthrough_opens += 1
            return None
        if path not in self.snapshot.contents:
            self.misses += 1
            raise ValueError(f"Metadata snapshot missing contents: {path}; recapture all splits")
        self.hits += 1
        stream = io.BytesIO(self.snapshot.contents[path])
        return stream if mode == "rb" else io.TextIOWrapper(stream)


@dataclass
class MetadataCapture:
    """Recorded constructor reads; sample payloads and binary indexes stay live."""

    recipe_identity: str
    contents: dict[str, bytes] = field(default_factory=dict)
    file_checks: dict[str, bool] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    jsonl_sizes: dict[str, int] = field(default_factory=dict)
    captured_bytes: int = 0
    calls: dict[str, int] = field(default_factory=dict)

    def record(self, kind: str, key: str, value: bool | int) -> None:
        entries = getattr(self, kind)
        if key in entries and entries[key] != value:
            raise ValueError(f"Dataset metadata changed during capture: {key}")
        entries[key] = value
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def open(self, path: str, mode: str, reader: Callable[[], BinaryIO]):
        if mode not in ("r", "rb") or not _metadata_path(path):
            return None
        self.calls["contents"] = self.calls.get("contents", 0) + 1
        if path not in self.contents:
            with reader() as stream:
                payload = stream.read(_MAX_BYTES - self.captured_bytes + 1)
            self.captured_bytes += len(payload)
            if self.captured_bytes > _MAX_BYTES:
                raise ValueError("Metadata capture exceeded 64 MiB; narrow the capture scope")
            self.contents[path] = payload
        stream = io.BytesIO(self.contents[path])
        return stream if mode == "rb" else io.TextIOWrapper(stream)

    def record_index_generations(self) -> None:
        from megatron.energon.epathlib import EPath

        paths = set(self.contents) | set(self.file_checks) | set(self.sizes)
        roots = {key.split("/.nv-meta/", 1)[0] for key in paths if "/.nv-meta/" in key}
        for root in sorted(roots):
            path = EPath(root) / ".nv-meta/index.uuid"
            self.record("file_checks", str(path), path.is_file())
            if self.file_checks[str(path)]:
                with self.open(str(path), "rb", lambda: path.open("rb")):
                    pass


_ACTIVE_SNAPSHOT: contextvars.ContextVar[tuple[int, MetadataReplay] | None] = (
    contextvars.ContextVar("energon_metadata_snapshot", default=None)
)
_ACTIVE_CAPTURE: contextvars.ContextVar[tuple[int, MetadataCapture] | None] = (
    contextvars.ContextVar("energon_metadata_capture", default=None)
)


def active_snapshot() -> MetadataReplay | None:
    entry = _ACTIVE_SNAPSHOT.get()
    return entry[1] if entry is not None and entry[0] == os.getpid() else None


def active_capture() -> MetadataCapture | None:
    entry = _ACTIVE_CAPTURE.get()
    return entry[1] if entry is not None and entry[0] == os.getpid() else None


@contextlib.contextmanager
def use_metadata_snapshot(validated: ValidatedMetadataSnapshot) -> Iterator[MetadataReplay]:
    """Replay validated metadata only in the constructing process, with strict coverage."""
    if not isinstance(validated, ValidatedMetadataSnapshot):
        raise TypeError("Call MetadataSnapshot.validate() before replay")
    if active_snapshot() is not None or active_capture() is not None:
        raise RuntimeError("Metadata capture/replay scopes cannot overlap")
    validated.snapshot.check_storage_root()
    replay = MetadataReplay(validated.snapshot)
    token = _ACTIVE_SNAPSHOT.set((os.getpid(), replay))
    try:
        yield replay
    finally:
        _ACTIVE_SNAPSHOT.reset(token)


@contextlib.contextmanager
def capture_metadata(output: str | Path, *, recipe_identity: str) -> Iterator[MetadataCapture]:
    """Capture constructor reads and atomically publish a snapshot and digest audit.

    The output directory must exist. The audit is named <complete filename>.audit.json.
    Neither output is published when construction fails. Capture training and
    validation construction in the same context, without starting iteration.
    """
    if active_snapshot() is not None or active_capture() is not None:
        raise RuntimeError("Metadata capture/replay scopes cannot overlap")
    output = Path(output)
    audit_path = output.with_name(output.name + ".audit.json")
    if output.is_dir() or audit_path.is_dir():
        raise IsADirectoryError("Snapshot and audit destinations must be files")
    storage_root = _storage_root()
    # Open both temporaries before construction to fail early on invalid destinations.
    with _temporary_output(output) as tmp, _temporary_output(audit_path) as audit:
        capture = MetadataCapture(recipe_identity)
        token = _ACTIVE_CAPTURE.set((os.getpid(), capture))
        try:
            yield capture
        finally:
            _ACTIVE_CAPTURE.reset(token)
        capture.record_index_generations()
        if storage_root != _storage_root():
            raise ValueError("NVDATASET_CACHE_DIR changed during capture")
        document = {
            "format_version": 2,
            "recipe_identity": recipe_identity,
            "storage_root": storage_root,
            "contents": {
                key: base64.b64encode(value).decode("ascii")
                for key, value in capture.contents.items()
            },
            "file_checks": capture.file_checks,
            "sizes": capture.sizes,
            "jsonl_sizes": capture.jsonl_sizes,
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        tmp.write(payload)
        tmp.flush()
        audit.write(
            json.dumps(
                {
                    "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                    "recipe_identity": recipe_identity,
                    "calls": capture.calls,
                    "captured_bytes": capture.captured_bytes,
                    "package_bytes": len(payload),
                },
                indent=2,
            ).encode()
            + b"\n"
        )
        audit.flush()
        os.replace(tmp.name, output)
        os.replace(audit.name, audit_path)


@contextlib.contextmanager
def _temporary_output(destination: Path) -> Iterator[BinaryIO]:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", delete=False
    ) as stream:
        try:
            yield stream
        finally:
            Path(stream.name).unlink(missing_ok=True)
