#!/bin/bash

git format-patch -2 --abbrev=40 --zero-commit --no-signature -- cmake include src CMakeLists.txt :!include/.styleguide :!src/.styleguide
