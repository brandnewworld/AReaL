# ps aux | grep areal
set -x

ACTOR_PATH=${ACTOR_PATH:-"Qwen/Qwen3-1.7B"}
BATCH_SIZE=${BATCH_SIZE:-32}
EXT_MODEL_NAME="${ACTOR_PATH##*/}"
GRPO_N_SAMPLES=${GRPO_N_SAMPLES:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
SGLANG_MEM_FRAC=${SGLANG_MEM_FRAC:-0.9}
TOTAL_TRAIN_EPOCHS=${TOTAL_TRAIN_EPOCHS:-2}
ALLOCATION_MODE=${ALLOCATION_MODE:-"vllm:d2t1+fsdp:d2"}

# actor.mb_spec.max_tokens_per_mb
# max_concurrent_rollouts
# sglang.mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity

PYTHONPATH=. python3 -m areal.launcher.local examples/math/deepmath_grpo.py \
    --config examples/math/deepmath_grpo.yaml \
    experiment_name="EXP_${EXT_MODEL_NAME}" \
    trial_name="E_B${BATCH_SIZE}" \
    allocation_mode="${ALLOCATION_MODE}" \
    actor.path="${ACTOR_PATH}" \
    async_training=True \
    ref.path="${ACTOR_PATH}" \
    total_train_epochs=${TOTAL_TRAIN_EPOCHS} \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=${N_GPUS_PER_NODE} \
    gconfig.n_samples=${GRPO_N_SAMPLES} \
    train_dataset.batch_size=${BATCH_SIZE} \
    train_dataset.max_length=${MAX_PROMPT_LENGTH} \
    stats_logger.wandb.mode=offline \
    sglang.mem_fraction_static=${SGLANG_MEM_FRAC} \
    +sglang.attention_backend=triton
