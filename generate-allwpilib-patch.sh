#!/bin/bash

git format-patch -10 --abbrev=40 --zero-commit --no-signature -- include src
