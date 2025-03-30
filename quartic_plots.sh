#!/bin/bash

set -e

./tools/run-release-test.sh 'Problem - Quartic' && ./plot_xy.py x_results.csv v_results.csv mu_results.csv
