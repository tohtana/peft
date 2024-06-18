NUM_NODES=$(wc -l < /job/hostfile)
LOCKFILE=/tmp/lockfile
DS_DIR=/scratch/amlt_code/DeepSpeed
CODE_DIR=/scratch/amlt_code/peft/examples/sft

echo "NUM_NODES: ${NUM_NODES}"

for i in $(seq 1 $((${NUM_NODES} - 1))); do
    rsync -av --exclude='.git' --exclude='__pycache__' -e ssh ${CODE_DIR}/ node-${i}:${CODE_DIR}/
    rsync -av --exclude='.git' --exclude='__pycache__' -e ssh ${DS_DIR}/ node-${i}:${DS_DIR}/
done

HOST_IP=$(hostname -i)
NUM_PROCESSES=$((NUM_NODES * 8))
ds_ssh "cd /scratch/amlt_code/peft/examples/sft; bash ./run_deepspeed_z3.sh ${HOST_IP} ${NUM_NODES} ${NUM_PROCESSES} 2>&1 | tee debug.log"
