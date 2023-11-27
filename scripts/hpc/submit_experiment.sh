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
  N_STEPS=5
  EPOCHS_PER_STEP=40
  SSL_TOTAL_EPOCHS=200  ### MUST BE N_STEPS x EPOCHS_PER_STEO
  let FINAL_STEP=$N_STEPS-1
  EXP_NAME="$1"
  CONF_PATH="$2"
  #CONF_PATH="/home/$(whoami)/experiments/PCG/configs/test_config.yml"
  # shellcheck disable=SC2006
  CODE_PATH="/git/listen2yourheart"
  CHK_PREFIX="/home/$(whoami)/experiments/PCG/runs/${EXP_NAME}"
  EXP_NUM=`find "${CHK_PREFIX}"/* -maxdepth 0 -type d | wc -l`
  LOGDIR="${CHK_PREFIX}/${EXP_NUM}/logs"
  SSL_PATH="${CHK_PREFIX}/${EXP_NUM}/ssl"
  DS_PATH="${CHK_PREFIX}/${EXP_NUM}/downstream"
  mkdir -p "${CHK_PREFIX}/${EXP_NUM}"
  mkdir -p "${LOGDIR}"
  mkdir -p "${SSL_PATH}"
  mkdir -p "${DS_PATH}"

  timestamp=$(date +%s)
  epochs=0
  CHK_NAME="${EXP_NAME}.chk"
  jid=${EXP_NAME}_${timestamp}_0
  STDOUT="${LOGDIR}/$jid"


  CHK_FILE="${CHK_PREFIX}/${EXP_NUM}/${CHK_NAME}"
  sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" $CODE_PATH/scripts/hpc/pretrain.sh --tmp_path $SSL_PATH --initial_epoch 0 --ssl_job_epochs $EPOCHS_PER_STEP --ssl_total_epochs $SSL_TOTAL_EPOCHS --conf_path $CONF_PATH
  echo $EPOCHS_PER_STEP > ${CHK_FILE}.numepochs

  for ((i=1; i<${N_STEPS}; i++)); do
      epochs=`cat ${CHK_FILE}.numepochs` # $epochs+$EPOCHS_PER_STEP
      let tot_epochs=$epochs+$EPOCHS_PER_STEP
      let d=$i-1
      jid=${EXP_NAME}_${timestamp}_$i
      STDOUT="${LOGDIR}/$jid"
      depends=$(squeue --noheader --format %i --name ${EXP_NAME}_${timestamp}_${d})
      sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" -d afterany:${depends} $CODE_PATH/scripts/hpc/pretrain.sh --tmp_path $SSL_PATH --initial_epoch $epochs --ssl_job_epochs $EPOCHS_PER_STEP --ssl_total_epochs $SSL_TOTAL_EPOCHS --conf_path $CONF_PATH
      PREV_FILE=${CHK_FILE}
      echo $tot_epochs > ${CHK_FILE}.numepochs
  done

  ds_jid=${EXP_NAME}_${timestamp}_${N_STEPS}
  STDOUT="${LOGDIR}/$ds_jid"
  depends=$(squeue --noheader --format %i --name ${EXP_NAME}_${timestamp}_${FINAL_STEP})
  sbatch -J $ds_jid -o "${STDOUT}.out" -e "${STDOUT}.err" -d afterany:${depends} $CODE_PATH/scripts/hpc/downstream.sh --ssl_path $SSL_PATH --ds_path $DS_PATH --conf_path $CONF_PATH
fi
