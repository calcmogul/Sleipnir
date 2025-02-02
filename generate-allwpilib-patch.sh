#!/bin/bash

git format-patch -8 --abbrev=40 --zero-commit --no-signature -- include src
