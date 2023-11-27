#!/bin/bash


# Docker configuration
export IMAGE_NAME=listen-to-your-heart
export CONTAINER_NAME=listen-to-your-heart


# Path configuration
dataset_path=/home/vasileios/tmp/physionet2022challenge/dataset
results_path=/home/vasileios/tmp/physionet2022challenge/results/docker

export TRAIN_SET_PATH="${dataset_path}/train"
export TEST_SET_PATH="${dataset_path}/test"
export MODEL_PATH="${results_path}/model"
export TEST_OUTPUTS_PATH="${results_path}/test_outputs"
