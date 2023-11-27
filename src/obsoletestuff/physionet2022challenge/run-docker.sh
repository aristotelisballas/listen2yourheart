#!/bin/bash


source env.sh


# Initialization
mkdir -p ${MODEL_PATH}
mkdir -p ${TEST_OUTPUTS_PATH}


# Main
docker run \
       -it \
       -v ${TRAIN_SET_PATH}:/physionet/training_data \
       -v ${TEST_SET_PATH}:/physionet/test_data \
       -v ${MODEL_PATH}:/physionet/model \
       -v ${TEST_OUTPUTS_PATH}:/physionet/test_outputs \
       --name ${CONTAINER_NAME} \
       --rm \
       --runtime=nvidia \
       ${IMAGE_NAME} \
       bash
