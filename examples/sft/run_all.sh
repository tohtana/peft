#!/bin/bash

# add option
# if --backend fsdp, run run_fsdp.sh, else run run_deepspeed_z3.sh
# if --compile, set COMPILE_DS=1
# Implement the options here



BACKEND=${1:-deepspeed}
MODEL=Meta-Llama-3-8B

SCRIPT=run_deepspeed_z3.sh
if [ "${BACKEND}" == "fsdp" ]; then
    SCRIPT=run_fsdp.sh
elif [ "${BACKEND}" != "deepspeed" ]; then
    echo "Invalid backend: ${BACKEND}"
    exit 1
fi

NUM_NODES=$(wc -l < /job/hostfile)
LOCKFILE=/tmp/lockfile
DS_DIR=/scratch/amlt_code/DeepSpeed
CODE_DIR=/scratch/amlt_code/peft/examples/sft
TRANSFORMERS_DIR=/scratch/amlt_code/transformers

NGPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_PROCESSES=$((NUM_NODES * NGPUS_PER_NODE))

echo "NUM_NODES: ${NUM_NODES}"
echo "NGPUS_PER_NODE: ${NGPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"

for i in $(seq 1 $((${NUM_NODES} - 1))); do
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='mlruns' --exclude='__runs__' -e ssh ${CODE_DIR}/ node-${i}:${CODE_DIR}/
    rsync -av --exclude='.git' --exclude='__pycache__' -e ssh ${DS_DIR}/ node-${i}:${DS_DIR}/
    rsync -av --exclude='.git' --exclude='__pycache__' -e ssh ${TRANSFORMERS_DIR}/ node-${i}:${TRANSFORMERS_DIR}/
done

HOST_IP=$(hostname -i)

ds_ssh "pkill -u aiscuser -f [a]ccelerate"
ds_ssh "cd /scratch/amlt_code/peft/examples/sft; bash ./${SCRIPT} ${HOST_IP} ${NUM_NODES} ${NUM_PROCESSES} ${MODEL} 2>&1 | tee debug_${BACKEND}.log"
