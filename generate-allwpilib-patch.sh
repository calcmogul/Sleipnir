#!/bin/bash

git format-patch -3 --abbrev=40 --zero-commit --no-signature -- include src
