#!/bin/bash
git format-patch -1 --abbrev=40 --zero-commit --no-signature -- include/ src/
