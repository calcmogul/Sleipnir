#!/bin/bash

git format-patch -7 --abbrev=40 --zero-commit --no-signature -- include src
