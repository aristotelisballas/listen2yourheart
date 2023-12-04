#!/bin/bash


# Configuration
DST=~/workspace/Datasets/ephnogram

# Main
mkdir -p "$DST"
cd "$DST" || exit

wget -r -N -c -np https://physionet.org/files/ephnogram/1.0.0/WFDB/
cd physionet.org/files/ephnogram/1.0.0 || exit
wget -N -c -np https://physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv
wget -N -c -np https://physionet.org/files/ephnogram/1.0.0/LICENSE.txt
wget -N -c -np https://physionet.org/files/ephnogram/1.0.0/RECORDS
wget -N -c -np https://physionet.org/files/ephnogram/1.0.0/SHA256SUMS.txt
