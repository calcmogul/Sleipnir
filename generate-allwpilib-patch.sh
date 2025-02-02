#!/bin/bash

git format-patch -9 --abbrev=40 --zero-commit --no-signature -- include src
