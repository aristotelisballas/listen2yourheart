FROM tensorflow/tensorflow:latest-gpu

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER cdiou@hua.gr

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ../../.. /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
