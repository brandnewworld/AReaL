# pkill -f "areal|vllm|VLLM::"; sleep 10; pkill -9 -f "areal|vllm|VLLM::"
# nohup ./run_qwen3_0520.sh > /dev/null 2>&1 &
#
# Multi-turn retool-dapo GRPO on AReaL, ported from the verl-0519 script
#   scripts/experiment_runs/bs32_n4_onpolicy_8gpu_lr5e6_minibs8_grpo.sh
# Learning-process parameters mirror that script; topology stays AReaL-native
# 4 GPU rollout (vllm, TP=1) + 4 GPU train (fsdp).
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

ACTOR_PATH="/external/InnovationLab/models/Qwen/Qwen3-4B"
ALLOCATION_MODE="vllm:d4t1+fsdp:d4"
AREAL_ROOT="/tmp/areal"
BATCH_SIZE=32
EVAL_FREQ_STEPS=5
GRPO_N_SAMPLES=4
LR=5e-6
MAX_NEW_TOKENS=16384
MAX_PROMPT_LENGTH=2048
MAX_TURNS=8
PPO_N_MINIBATCHES=4
TOTAL_TRAIN_EPOCHS=1
TOTAL_TRAIN_STEPS=100
TRAIN_DATASET_PATH="/external/InnovationLab/data/retool-dapo-math-scout-toolmandate/train.parquet"
VALID_DATASET_PATH="/external/InnovationLab/data/retool-dapo-math-scout-toolmandate/test_mini128.parquet"
VLLM_MEM_UTIL=0.85

################ 2 ################

ACTOR_PATH=${ACTOR_PATH:-"/external/InnovationLab/models/Qwen/Qwen3-4B"}
ALLOCATION_MODE=${ALLOCATION_MODE:-"vllm:d4t1+fsdp:d4"}
AREAL_ROOT=${AREAL_ROOT:-"${DEV_ROOT_PATH}/tmp/areal"}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_FREQ_STEPS=${EVAL_FREQ_STEPS:-5}
GRPO_N_SAMPLES=${GRPO_N_SAMPLES:-4}
LR=${LR:-5e-6}
MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS:-$(( BATCH_SIZE * GRPO_N_SAMPLES ))}
MAX_HEAD_OFFPOLICYNESS=${MAX_HEAD_OFFPOLICYNESS:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_TURNS=${MAX_TURNS:-8}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-$(( MAX_PROMPT_LENGTH + MAX_NEW_TOKENS ))}
PPO_N_MINIBATCHES=${PPO_N_MINIBATCHES:-4}
SGLANG_MEM_FRAC=${SGLANG_MEM_FRAC:-0.9}
TOTAL_TRAIN_EPOCHS=${TOTAL_TRAIN_EPOCHS:-1}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-100}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"/external/InnovationLab/data/retool-dapo-math-scout-toolmandate/train.parquet"}
VALID_DATASET_PATH=${VALID_DATASET_PATH:-"/external/InnovationLab/data/retool-dapo-math-scout-toolmandate/test_mini128.parquet"}
VLLM_MEM_UTIL=${VLLM_MEM_UTIL:-0.85}

################ 3 ################

EXT_MODEL_NAME="${ACTOR_PATH##*/}"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARNING

PYTHONPATH=. python3 -m areal.launcher.local examples/math/retool_grpo.py \
    --config examples/math/retool_grpo.yaml \
    actor.path="${ACTOR_PATH}" \
    actor.optimizer.lr=${LR} \
    actor.ppo_n_minibatches=${PPO_N_MINIBATCHES} \
    allocation_mode="${ALLOCATION_MODE}" \
    async_training=True \
    cluster.fileroot="${AREAL_ROOT}/experiments" \
    cluster.n_gpus_per_node=${N_GPUS_PER_NODE} \
    cluster.n_nodes=1 \
    evaluator.freq_steps=${EVAL_FREQ_STEPS} \
    experiment_name="EXP_RETOOL_${EXT_MODEL_NAME}" \
    gconfig.max_new_tokens=${MAX_NEW_TOKENS} \
    gconfig.n_samples=${GRPO_N_SAMPLES} \
    ref.path="${ACTOR_PATH}" \
    retool.max_turns=${MAX_TURNS} \
    retool.max_total_tokens=${MAX_TOTAL_TOKENS} \
    rollout.max_head_offpolicyness=${MAX_HEAD_OFFPOLICYNESS} \
    rollout.max_concurrent_rollouts=${MAX_CONCURRENT_ROLLOUTS} \
    total_train_epochs=${TOTAL_TRAIN_EPOCHS} \
    total_train_steps=${TOTAL_TRAIN_STEPS} \
    train_dataset.batch_size=${BATCH_SIZE} \
    train_dataset.max_length=${MAX_PROMPT_LENGTH} \
    train_dataset.path=${TRAIN_DATASET_PATH} \
    trial_name="E${CURRENT_TIME}_B${BATCH_SIZE}" \
    +sglang.attention_backend=triton \
    sglang.mem_fraction_static=${SGLANG_MEM_FRAC} \
    stats_logger.wandb.mode=offline \
    valid_dataset.batch_size=${BATCH_SIZE} \
    valid_dataset.max_length=${MAX_PROMPT_LENGTH} \
    valid_dataset.path=${VALID_DATASET_PATH} \
    vllm.gpu_memory_utilization=${VLLM_MEM_UTIL}

notify "End of E${CURRENT_TIME}"
