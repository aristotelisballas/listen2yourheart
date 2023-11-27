#!/bin/bash


# To run this script, this is required:
#     pip install pipreqs pip-tools

rm requirements.*
pipreqs --savepath=requirements-raw.in
grep -v "absl==0.0" requirements-raw.in > requirements.in
pip-compile # --resolver=backtracking

rm requirements*.in
