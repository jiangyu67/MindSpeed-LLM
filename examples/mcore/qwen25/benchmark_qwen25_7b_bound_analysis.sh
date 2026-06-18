#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
# Ascend env scripts may read shell variables like ZSH_VERSION without default guards.
# Temporarily disable nounset to avoid unbound variable failures while sourcing.
set +u
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export PYTHONPATH=/home/MindSpeed:/home/MindSpeed-LLM:${PYTHONPATH:-}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_DIR=$(cd "${PROJECT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

NPUS_PER_NODE=${NPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6010}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

TRAIN_ITERS=${TRAIN_ITERS:-100}
PROFILE_STEP_START=${PROFILE_STEP_START:-30}
PROFILE_STEP_END=${PROFILE_STEP_END:-31}
NPU_SMI_INTERVAL_SEC=${NPU_SMI_INTERVAL_SEC:-5}

CKPT_LOAD_DIR=${CKPT_LOAD_DIR:-"${WORKSPACE_DIR}/model_weights/qwen2.5_mcore"}
CKPT_SAVE_DIR=${CKPT_SAVE_DIR:-"${WORKSPACE_DIR}/ckpt/qwen25-7b"}
DATA_PATH=${DATA_PATH:-"${WORKSPACE_DIR}/dataset/alpaca_text_document"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${WORKSPACE_DIR}/model_from_hf/qwen2.5-7b-hf"}

TP=${TP:-1}
PP=${PP:-4}
GBS=${GBS:-64}

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${WORKSPACE_DIR}/logs/bound_analysis/${TS}"
mkdir -p "${OUT_DIR}"

MANIFEST_CSV="${OUT_DIR}/manifest.csv"
echo "case_name,seq_len,mbs,use_fused_swiglu,log_file,npu_file,profile_dir,exit_code" > "${MANIFEST_CSV}"

DISTRIBUTED_ARGS="
  --nproc_per_node ${NPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT}
"

start_npu_sampler() {
  local output_file="$1"
  (
    while true; do
      echo "===== $(date '+%F %T') ====="
      npu-smi info || true
      sleep "${NPU_SMI_INTERVAL_SEC}"
    done
  ) > "${output_file}" 2>&1 &
  echo $!
}

append_manifest() {
  local case_name="$1"
  local seq_len="$2"
  local mbs="$3"
  local use_fused_swiglu="$4"
  local log_file="$5"
  local npu_file="$6"
  local profile_dir="$7"
  local exit_code="$8"
  echo "${case_name},${seq_len},${mbs},${use_fused_swiglu},${log_file},${npu_file},${profile_dir},${exit_code}" >> "${MANIFEST_CSV}"
}

run_case() {
  local case_name="$1"
  local seq_len="$2"
  local mbs="$3"
  local use_fused_swiglu="$4"

  local case_dir="${OUT_DIR}/${case_name}"
  local log_file="${case_dir}/train.log"
  local npu_file="${case_dir}/npu_smi.log"
  local profile_dir="${case_dir}/profile"
  mkdir -p "${case_dir}" "${profile_dir}"

  local maybe_fused=""
  if [[ "${use_fused_swiglu}" == "1" ]]; then
    maybe_fused="--use-fused-swiglu"
  fi

  local gpt_args="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --max-position-embeddings ${seq_len} \
    --seq-length ${seq_len} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    ${maybe_fused} \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${mbs} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --train-iters ${TRAIN_ITERS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16 \
    --profile \
    --profile-ranks 0 \
    --profile-level level1 \
    --profile-step-start ${PROFILE_STEP_START} \
    --profile-step-end ${PROFILE_STEP_END} \
    --profile-save-path ${profile_dir}
  "

  local data_args="
    --data-path ${DATA_PATH} \
    --split 100,0,0
  "

  local ckpt_args="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    --save ${CKPT_SAVE_DIR}
  "

  local output_args="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --log-throughput
  "

  echo "[RUN] ${case_name} seq_len=${seq_len} mbs=${mbs} fused_swiglu=${use_fused_swiglu}"
  
  # Check if msprof is available and requested
  # For simplicity, let's enable msprof for specific cases or globally if available
  # However, msprof requires wrapping the command.
  local use_msprof=0
  
  if command -v msprof >/dev/null 2>&1; then
      use_msprof=1
  fi
  
  local sampler_pid=""
  if [[ "${use_msprof}" == "0" ]]; then
      sampler_pid=$(start_npu_sampler "${npu_file}")
  fi

  set +e
  if [[ "${use_msprof}" == "1" ]]; then
      echo "[INFO] Running with msprof..."
      # Scheme B: Write the complex command to a temporary script to avoid argv length limit
      local wrapper_script="${case_dir}/run_wrapper.sh"
      
      # Create the wrapper script content
      # IMPORTANT: Variables are expanded by the parent shell when creating the file.
      # We need to make sure the arguments are correctly formatted on a single line.
      cat <<EOF > "${wrapper_script}"
#!/bin/bash
set -e
# Export current environment variables to the wrapper script
export PYTHONPATH="${PYTHONPATH}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT}"

# Flatten arguments to avoid newline issues
# We use eval to handle potential quoting issues in the variable expansion, although simple expansion is usually safer here.
# But since the previous attempt failed with "command not found" for arguments starting with --, it means
# the shell tried to execute "--use-mcore-models" as a command!
# This happens if there's a newline before it.

# Let's constructing the full command line carefully.
CMD="python3.10 -m torch.distributed.run ${DISTRIBUTED_ARGS} pretrain_gpt.py ${gpt_args} ${data_args} ${ckpt_args} ${output_args} --distributed-backend nccl --transformer-impl local"

# Execute the command
exec \$CMD
EOF
      chmod +x "${wrapper_script}"
      
      # Run msprof pointing to the wrapper script
      # Note: msprof still needs --application to point to the script
      msprof --output="${profile_dir}" --application="${wrapper_script}" 2>&1 | tee "${log_file}"
      
      # Clean up wrapper script (optional, maybe keep for debugging)
      # rm "${wrapper_script}"
  else
      python3.10 -m torch.distributed.run ${DISTRIBUTED_ARGS} pretrain_gpt.py \
        ${gpt_args} \
        ${data_args} \
        ${ckpt_args} \
        ${output_args} \
        --distributed-backend nccl \
        --transformer-impl local \
        2>&1 | tee "${log_file}"
  fi
  local exit_code=${PIPESTATUS[0]}
  set -e

  if [[ -n "${sampler_pid}" ]]; then
      kill "${sampler_pid}" >/dev/null 2>&1 || true
      wait "${sampler_pid}" 2>/dev/null || true
  fi

  # Auto-retry logic for High MBS
  if [[ "${exit_code}" != "0" ]] && [[ "${case_name}" == "high_mbs" ]] && [[ "${mbs}" -gt 1 ]]; then
      echo "[WARN] high_mbs (MBS=${mbs}) failed. Retrying with MBS=$((mbs-1))..."
      run_case "high_mbs_retry_mbs$((mbs-1))" "${seq_len}" "$((mbs-1))" "${use_fused_swiglu}"
      return
  fi

  append_manifest "${case_name}" "${seq_len}" "${mbs}" "${use_fused_swiglu}" "${log_file}" "${npu_file}" "${profile_dir}" "${exit_code}"

  if [[ "${exit_code}" != "0" ]]; then
    echo "[WARN] ${case_name} failed with exit_code=${exit_code}"
  fi
}

# Wrapper for msprof
run_msprof_case() {
    local case_name="$1"
    # msprof command construction if needed, but for now we stick to the requested changes in run_case logic
    # The user asked to "Switch to msprof" which implies changing how the command is invoked.
    # However, msprof is usually a wrapper around the python command.
    # Let's adjust run_case to use msprof when a flag is set or by default.
    :
}


echo "[INFO] Start bound analysis benchmark at $(date '+%F %T')"
echo "[INFO] Output directory: ${OUT_DIR}"

run_case baseline 4096 1 1
run_case low_seq 2048 1 1
run_case high_mbs 4096 2 1
run_case no_fused_swiglu 4096 1 0

echo "[INFO] Start analysis"
python3.10 "${SCRIPT_DIR}/analyze_qwen25_bound_results.py" \
  --manifest "${MANIFEST_CSV}" \
  --output-dir "${OUT_DIR}"

echo "[DONE] Bound analysis finished. See: ${OUT_DIR}"
