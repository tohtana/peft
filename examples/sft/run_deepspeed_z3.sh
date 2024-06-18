HOST_IP=$1
NUM_NODES=$2
NUM_PROCESSES=$3

export NCCL_DEBUG=WARN

echo "HOST_IP: ${HOST_IP}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"

MACHINE_RANK=$(hostname | sed 's/[^0-9]*//g')

cat configs/deepspeed_config_z3.yaml.template \
| sed "s/__MACHINE_RANK__/${MACHINE_RANK}/g" \
| sed "s/__NUM_MACHINES__/${NUM_NODES}/g" \
| sed "s/__NUM_PROCESSES__/${NUM_PROCESSES}/g" \
> configs/deepspeed_config_z3.yaml

/home/aiscuser/.local/bin/accelerate launch --main_process_ip ${HOST_IP} --main_process_port 12345 \
--num_machines ${NUM_NODES} --num_processes ${NUM_PROCESSES} --machine_rank ${MACHINE_RANK} \
--config_file "configs/deepspeed_config_z3.yaml"  train.py \
--seed 100 \
--model_name_or_path "/blob/projects/ds_rlhf/models/Meta-Llama-3-70B" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-qlora-dsz3" \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant True \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora False \
--use_4bit_quantization False \
--use_nested_quant False \
