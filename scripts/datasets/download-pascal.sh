#!/bin/bash


# Configration
DST=~/workspace/Datasets/pascal

# Main
mkdir -p "$DST"
cd "$DST" || exit

# List of zip files
zipfiles=(
  "Atraining_normal.zip"
  "Atraining_murmur.zip"
  "Atraining_extrahs.zip"
  "Atraining_artifact.zip"
  "Aunlabelledtest.zip"
  "Btraining_normal.zip"
  "Btraining_murmur.zip"
  "Btraining_extrasystole.zip"
  "Bunlabelledtest.zip"
  )

for i in "${zipfiles[@]}"; do
  wget "http://www.peterjbentley.com/heartchallenge/wav/$i"
  unzip -o "$i"
  rm "$i"
done

# Special fixes
mv "Training B Normal" "Btraining_normal"
mv "Btraining_murmur/Btraining_noisymurmur" ./
mv "Btraining_normal/Btraining_noisynormal" ./

rm -rf __MACOSX
