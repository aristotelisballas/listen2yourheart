#!/bin/bash


# Configuration
DST=~/workspace/Datasets/SUFHSDB

# Main
mkdir -p "$DST"
cd "$DST" || exit

wget -r -N -c -np https://physionet.org/files/sufhsdb/1.0.1/
