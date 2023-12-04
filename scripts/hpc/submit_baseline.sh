#!/bin/bash

# Instructions: Call without arguments to start training from the
# beginning. Call with a single argument which will be the checkpoint
# file to start from a previously saved checkpoint.
#
# If checkfile.chk is the checkpoint filename, then number of
# completed epochs is stored in a file named checkfile.numepochs
#
# Please define the `EXP_NAME` variable in order to correspond
# to the name of the running experiment.
#
# The N_STEPS variable is the number of times the experiment will be run.
# The EPOCHS_PER_STEP is the epochs that the model will be trained in each
# experiment. Both the EPOCHS_PER_STEP var is passed to the .py script
# as an argument. The example .py script handles the rest.
#
# For example: If I want to train my model for 30 epochs and
# train my model for 3 epochs each run, I will set:
# N_STEPS = 10 , EPOCHS_PER_STEP=3

if [ -z $1 ]; then
  echo "Please specify experiment name and config.yml. With that order."
else
#  N_STEPS=5
#  EPOCHS_PER_STEP=40
#  let FINAL_STEP=$N_STEPS-1
  EXP_NAME="$1"
  CONF_PATH="$2"
  #CONF_PATH="/home/$(whoami)/experiments/PCG/configs/test_config.yml"
  # shellcheck disable=SC2006
  CHK_PREFIX="/home/$(whoami)/experiments/PCG/runs/${EXP_NAME}"
  EXP_NUM=`find "${CHK_PREFIX}"/* -maxdepth 0 -type d | wc -l`
  LOGDIR="${CHK_PREFIX}/${EXP_NUM}/logs"
#  SSL_PATH="${CHK_PREFIX}/${EXP_NUM}/ssl"
  DS_PATH="${CHK_PREFIX}/${EXP_NUM}/downstream"
  CODE_PATH="git/listen2yourheart"
  mkdir -p "${CHK_PREFIX}/${EXP_NUM}"
  mkdir -p "${LOGDIR}"
  mkdir -p "${DS_PATH}"

#  epochs=0
#  CHK_NAME="${EXP_NAME}.chk"
  jid=${EXP_NAME}_0
  STDOUT="${LOGDIR}/$jid"

#  ds_jid=${EXP_NAME}_0
  STDOUT="${LOGDIR}/$jid"
#  depends=$(squeue --noheader --format %i --name ${EXP_NAME}_${FINAL_STEP})
  sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" $CODE_PATH/scripts/hpc/baseline.sh  --ds_path $DS_PATH --conf_path $CONF_PATH
fi

