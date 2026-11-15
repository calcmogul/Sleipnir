#!/usr/bin/env python3

# Copyright (c) Sleipnir contributors

"""Copies the C++ library's sources into cpp-src/ so the crate can build
outside the repository, e.g., after `cargo publish`."""

import shutil
from pathlib import Path

CRATE_DIR = Path(__file__).resolve().parent
REPO_DIR = CRATE_DIR.parent.parent
DEST_DIR = CRATE_DIR / "cpp-src"

# Everything the root CMakeLists.txt reads with the Rust build's options
ENTRIES = [
    "CMakeLists.txt",
    "SleipnirConfig.cmake.in",
    "LICENSE.txt",
    "cmake",
    "include",
    "src",
]


def main():
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir()

    for entry in ENTRIES:
        source = REPO_DIR / entry
        if source.is_dir():
            shutil.copytree(source, DEST_DIR / entry)
        else:
            shutil.copy2(source, DEST_DIR / entry)

    print(f"Vendored the C++ sources into {DEST_DIR}")


if __name__ == "__main__":
    main()
