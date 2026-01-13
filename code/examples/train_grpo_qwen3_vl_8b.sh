#开启回显
set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the code directory (parent of examples)
CODE_DIR="$(dirname "$SCRIPT_DIR")"

# Add verl to PYTHONPATH BEFORE changing directory
# This ensures Python can find the verl module
export PYTHONPATH="${CODE_DIR}/verl:${CODE_DIR}:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Change to code directory so relative paths work correctly
cd "$CODE_DIR"
echo "Working directory: $(pwd)"

# Verify the module exists
if [ ! -f "${CODE_DIR}/verl/verl/trainer/main_ppo_vrag.py" ]; then
    echo "ERROR: main_ppo_vrag.py not found at ${CODE_DIR}/verl/verl/trainer/main_ppo_vrag.py"
    exit 1
fi
echo "Found main_ppo_vrag.py at ${CODE_DIR}/verl/verl/trainer/main_ppo_vrag.py"

#export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_V1=1
export RAY_memory_usage_threshold=0.995
ENGINE=${1:-vllm}
# 设置可见GPU，默认使用所有GPU，可通过环境变量覆盖
# 用法: CUDA_VISIBLE_DEVICES=2,3,4,5 ./train_grpo_qwen3_vl_8b.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# 根据CUDA_VISIBLE_DEVICES计算实际可用GPU数量
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Qwen3-VL-8B-Instruct 模型路径
model_path=/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/models/Qwen3-VL-8B-Thinking
#model_path=/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/models/Qwen3-VL-8B-Instruct
#model_path=/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/models/Qwen2.5-VL-7B-Instruct

train_batch_size=8
ppo_mini_batch_size=$((4 * n_gpus))
# 确保 train_batch_size >= ppo_mini_batch_size
if [ $ppo_mini_batch_size -gt $train_batch_size ]; then
    ppo_mini_batch_size=$train_batch_size
fi
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=2

# Memory optimization settings for 4x80GB GPUs
# Enable optimizer offload to reduce GPU memory usage during backward pass
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
# Reduce vLLM GPU memory utilization to leave more room for training
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}

tensor_model_parallel_size=1
val_before_train=False
search_url="http://0.0.0.0:8002/search"
#最大工具调用轮数
max_turns=16
project_name="vrag"
experiment_name="GRPO_w_crop_${n_gpus}_gpus_${max_turns}_maxturns_${n_agent}_ngroups_qwen3_vl_8b"

# Reward Model Configuration
# Local Skywork-Reward model for reward computation
USE_LOCAL_RM=${USE_LOCAL_RM:-true}
LOCAL_RM_PATH=${LOCAL_RM_PATH:-"/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/models/Skywork-Reward-V2-Llama-3.2-3B"}
LOCAL_RM_DEVICE=${LOCAL_RM_DEVICE:-"cpu"}  # Use CPU to avoid GPU memory conflicts


# Run the script directly to avoid conflicts with installed verl package
python3 "${CODE_DIR}/verl/verl/trainer/main_ppo_vrag.py" \
    algorithm.adv_estimator=grpo \
    data.train_files=/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/data/vidoseek_bak_train.parquet \
    data.val_files=/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/data/mmlongdoc_test.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=28672 \
    data.max_response_length=16384 \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=65536 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    ++actor_rollout_ref.rollout.n_agent=$n_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    ++reward_model.reward_manager='rm' \
    ++reward_model.use_local_rm=$USE_LOCAL_RM \
    ++reward_model.local_rm_path=$LOCAL_RM_PATH \
    ++reward_model.local_rm_device=$LOCAL_RM_DEVICE \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=3 \
    trainer.resume_mode=disable \
    trainer.val_before_train=$val_before_train \
    ++retriever.url=$search_url \
    ++max_turns=$max_turns $@
