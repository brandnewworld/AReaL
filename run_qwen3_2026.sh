# ps aux | grep areal
set -x

ACTOR_PATH=${ACTOR_PATH:-"Qwen/Qwen3-1.7B"}
BATCH_SIZE=${BATCH_SIZE:-32}
EXT_MODEL_NAME="${ACTOR_PATH##*/}"
GRPO_N_SAMPLES=${GRPO_N_SAMPLES:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}

PYTHONPATH=. python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    experiment_name="EXP_${EXT_MODEL_NAME}" \
    trial_name="E_B${BATCH_SIZE}" \
    allocation_mode=sglang.d2p1t1+d2p1t1 \
    actor.path="${ACTOR_PATH}" \
    ref.path="${ACTOR_PATH}" \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=${N_GPUS_PER_NODE} \
    gconfig.n_samples=${GRPO_N_SAMPLES} \
    gconfig.max_new_tokens=${MAX_NEW_TOKENS} \
    train_dataset.batch_size=${BATCH_SIZE} \
    launcher.inference_server_cpus_per_gpu=3 \
    launcher.inference_server_mem_per_gpu=28672 \
    launcher.trainer_cpus_per_gpu=3 \
    launcher.trainer_mem_per_gpu=28672 \
    stats_logger.wandb.mode=offline \
    +sglang.attention_backend=triton
