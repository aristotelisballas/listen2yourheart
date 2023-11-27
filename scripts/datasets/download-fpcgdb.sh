#!/bin/bash


# Configuration
DST=~/workspace/Datasets/fpcgdb

# Main
mkdir -p "$DST"
cd "$DST" || exit

wget -r -N -c -np https://physionet.org/files/fpcgdb/1.0.0/
