# pkill -f "areal|vllm|VLLM::"; sleep 10; pkill -9 -f "areal|vllm|VLLM::"
# nohup ./run_qwen3_2026.sh > /dev/null 2>&1 &
DEV_ROOT_PATH=${DEV_ROOT_PATH:-"$HOME"}
#DEV_ROOT_PATH=${DEV_ROOT_PATH:-"/external"}
CURRENT_TIME=${CURRENT_TIME:-$(TZ='UTC-8' date +%Y%m%d%H%M%S)}
LOG_FILE="R${CURRENT_TIME}.log"
exec > "${LOG_FILE}" 2>&1
set -x

NOTIFY_SCRIPT="notify_me.sh"
notify() {
    if [ -f "$NOTIFY_SCRIPT" ]; then
        bash "$NOTIFY_SCRIPT" "$1"
    fi
}

################ 1 ################

ACTOR_PATH="Qwen/Qwen3-1.7B"
#ALLOCATION_MODE="vllm:d1t1+fsdp:d1"
ALLOCATION_MODE="vllm:d2t1+fsdp:d2"
#ALLOCATION_MODE="'vllm:d2t1|fsdp:d2'"
AREAL_ROOT="${DEV_ROOT_PATH}/tmp/areal"
BATCH_SIZE=32
GRPO_N_SAMPLES=8
MAX_NEW_TOKENS=8192
TOTAL_TRAIN_EPOCHS=1
#TRAIN_DATASET_PATH="${DEV_ROOT_PATH}/data/deepmath-10.3k-debug"
VLLM_MEM_UTIL=0.9

################ 2 ################

ACTOR_PATH=${ACTOR_PATH:-"Qwen/Qwen3-1.7B"}
ALLOCATION_MODE=${ALLOCATION_MODE:-"vllm:d2t1+fsdp:d2"}
AREAL_ROOT=${AREAL_ROOT:-"${DEV_ROOT_PATH}/tmp/areal"}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_FREQ_STEPS=${EVAL_FREQ_STEPS:-"null"}
GRPO_N_SAMPLES=${GRPO_N_SAMPLES:-8}
MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS:-$(( BATCH_SIZE * GRPO_N_SAMPLES ))}
MAX_HEAD_OFFPOLICYNESS=${MAX_HEAD_OFFPOLICYNESS:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
SGLANG_MEM_FRAC=${SGLANG_MEM_FRAC:-0.9}
TOTAL_TRAIN_EPOCHS=${TOTAL_TRAIN_EPOCHS:-2}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${DEV_ROOT_PATH}/data/deepmath-10.3k"}
VALID_DATASET_PATH=${VALID_DATASET_PATH:-${TRAIN_DATASET_PATH}}
VLLM_MEM_UTIL=${VLLM_MEM_UTIL:-0.9}

################ 3 ################

EXT_MODEL_NAME="${ACTOR_PATH##*/}"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
export HF_HUB_OFFLINE=1

# actor.mb_spec.max_tokens_per_mb
# max_concurrent_rollouts
# sglang.mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity

PYTHONPATH=. python3 -m areal.launcher.local examples/math/deepmath_grpo.py \
    --config examples/math/deepmath_grpo.yaml \
    actor.path="${ACTOR_PATH}" \
    allocation_mode="${ALLOCATION_MODE}" \
    async_training=True \
    cluster.fileroot="${AREAL_ROOT}/experiments" \
    cluster.n_gpus_per_node=${N_GPUS_PER_NODE} \
    cluster.n_nodes=1 \
    evaluator.freq_steps=${EVAL_FREQ_STEPS} \
    experiment_name="EXP_${EXT_MODEL_NAME}" \
    gconfig.max_new_tokens=${MAX_NEW_TOKENS} \
    gconfig.n_samples=${GRPO_N_SAMPLES} \
    ref.path="${ACTOR_PATH}" \
    rollout.max_head_offpolicyness=${MAX_HEAD_OFFPOLICYNESS} \
    rollout.max_concurrent_rollouts=${MAX_CONCURRENT_ROLLOUTS} \
    total_train_epochs=${TOTAL_TRAIN_EPOCHS} \
    train_dataset.batch_size=${BATCH_SIZE} \
    train_dataset.max_length=${MAX_PROMPT_LENGTH} \
    train_dataset.path=${TRAIN_DATASET_PATH} \
    trial_name="E${CURRENT_TIME}_B${BATCH_SIZE}" \
    +sglang.attention_backend=triton \
    sglang.mem_fraction_static=${SGLANG_MEM_FRAC} \
    stats_logger.wandb.mode=offline \
    valid_dataset.path=${VALID_DATASET_PATH} \
    vllm.gpu_memory_utilization=${VLLM_MEM_UTIL}

notify "End of E${CURRENT_TIME}"
