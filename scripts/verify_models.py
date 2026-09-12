#!/usr/bin/env python3
"""Verify DreamX-Creator model files against the checked-in manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "model_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", nargs="?", type=Path, default=REPO / "checkpoints")
    parser.add_argument("--fast", action="store_true", help="check existence and sizes only")
    args = parser.parse_args()
    root = args.model_root.resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    total = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"MISSING  {entry['path']}")
            continue
        actual_size = path.stat().st_size
        total += actual_size
        if actual_size != entry["size"]:
            failures.append(
                f"SIZE     {entry['path']}: expected {entry['size']}, got {actual_size}"
            )
            continue
        if not args.fast:
            actual_hash = sha256(path)
            if actual_hash != entry["sha256"]:
                failures.append(
                    f"SHA256   {entry['path']}: expected {entry['sha256']}, got {actual_hash}"
                )
        print(f"OK       {entry['path']}")

    if failures:
        print("\nVerification failed:")
        print("\n".join(failures))
        return 1
    mode = "size" if args.fast else "SHA-256"
    print(f"\nVerified {len(manifest['files'])} files ({total / 1_000_000_000:.2f} GB) by {mode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
