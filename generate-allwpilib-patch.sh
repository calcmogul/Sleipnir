#!/bin/bash

git format-patch -6 --abbrev=40 --zero-commit --no-signature -- include src
