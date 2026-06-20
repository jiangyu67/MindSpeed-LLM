#!/bin/bash
# =============================================================================
# 下游任务评测脚本：bf16 baseline × 5 个任务 (ARC-C, ARC-E, BoolQ, LAMBADA, PIQA)
#
# 流程：
#   1. 将 Megatron checkpoint 转换为 HF 格式
#   2. 安装 lm-eval-harness（若未安装）
#   3. 对 HF 模型跑 5 个下游任务
#   4. 汇总结果
#
# 用法：
#   bash /home/zs/MindSpeed-LLM/examples/mcore/qwen25/eval_downstream_baseline.sh
#
# 可选环境变量：
#   SKIP_CONVERT=1   跳过转换步骤（已转换过时使用）
#   SKIP_INSTALL=1   跳过 lm-eval 安装
# =============================================================================

set -euo pipefail

# ---- 环境配置 ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONPATH=/home/zs/MindSpeed:/home/zs/MindSpeed-LLM:$PYTHONPATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# 使用 conda 环境的 libstdc++（系统的太旧，缺 GLIBCXX_3.4.29）
export LD_LIBRARY_PATH=/root/miniconda3/envs/qwen/lib:$LD_LIBRARY_PATH
# 让 lm-eval 把 cuda 映射到 NPU
export TORCH_NPU_AS_CUDA=1

# NPU 卡设置（使用 12,13,14,15）
export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15

PYTHON=/root/miniconda3/envs/qwen/bin/python3.10
PIP="/root/miniconda3/envs/qwen/bin/pip3.10"
REPO_DIR=/home/zs/MindSpeed-LLM
HF_REF_DIR=/home/zs/model_from_hf/qwen2.5-7b-hf
OUT_ROOT=/home/zs/ckpt/eval_downstream
LOG_DIR=${OUT_ROOT}/logs
mkdir -p "$LOG_DIR"

# ---- baseline 模型 checkpoint 配置 ----
MODEL_NAME="bf16"
MG_DIR=/home/zs/ckpt/qwen25-7b-bf16-baseline
HF_DIR=${OUT_ROOT}/hf_bf16

# ---- 5 个下游任务（lm-eval-harness 任务名）----
TASKS="arc_challenge,arc_easy,boolq,lambada_openai,piqa"

# 模型结构参数（与训练脚本一致）
TP=2
PP=1
NUM_LAYERS=28

# =============================================================================
# Step 0: 安装 lm-eval-harness
# =============================================================================
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    if ! $PYTHON -c "import lm_eval" 2>/dev/null; then
        echo "========== 安装 lm-eval-harness =========="
        $PIP install lm-eval 2>&1 | tail -5
    else
        echo "lm-eval-harness 已安装，跳过。"
    fi
fi

# =============================================================================
# Step 1: Megatron -> HF 转换
# =============================================================================
get_latest_iter() {
    local ckpt_dir="$1"
    local latest_file="${ckpt_dir}/latest_checkpointed_iteration.txt"
    if [ -f "$latest_file" ]; then
        cat "$latest_file"
    else
        echo ""
    fi
}

if [ "${SKIP_CONVERT:-0}" != "1" ]; then
    echo ""
    echo "========== Step 1: Megatron -> HF 转换 =========="
    echo "---- 转换 [${MODEL_NAME}]: ${MG_DIR} -> ${HF_DIR} ----"

    if [ ! -d "$MG_DIR" ]; then
        echo "[错误] checkpoint 目录不存在: ${MG_DIR}"
        exit 1
    fi

    iter=$(get_latest_iter "$MG_DIR")
    if [ -z "$iter" ]; then
        echo "[警告] 未找到 latest_checkpointed_iteration.txt，尝试用 release。"
        iter="release"
    fi
    echo "  使用迭代: ${iter}"

    mkdir -p "$HF_DIR"
    cp -n "$HF_REF_DIR"/tokenizer*.json "$HF_DIR"/ 2>/dev/null || true
    cp -n "$HF_REF_DIR"/vocab.json "$HF_DIR"/ 2>/dev/null || true
    cp -n "$HF_REF_DIR"/merges.txt "$HF_DIR"/ 2>/dev/null || true
    cp -n "$HF_REF_DIR"/tokenizer_config.json "$HF_DIR"/ 2>/dev/null || true
    cp -n "$HF_REF_DIR"/config.json "$HF_DIR"/ 2>/dev/null || true
    cp -n "$HF_REF_DIR"/generation_config.json "$HF_DIR"/ 2>/dev/null || true

    cd "$REPO_DIR"
    $PYTHON mindspeed_llm/tasks/checkpoint/convert_param.py \
        --cvt-type mg2hf \
        --model-name llama \
        --mg-dir "$MG_DIR" \
        --hf-dir "$HF_DIR" \
        --model-config-file "$HF_REF_DIR/config.json" \
        --model-index-file "$HF_REF_DIR/model.safetensors.index.json" \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --num-layers $NUM_LAYERS \
        --make-vocab-size-divisible-by 1 \
        --iteration "$iter" \
        2>&1 | tee "${LOG_DIR}/convert_${MODEL_NAME}.log"

    # 拷贝 safetensors 索引文件（转换脚本不生成，transformers 加载分片权重需要）
    cp -n "$HF_REF_DIR"/model.safetensors.index.json "$HF_DIR"/ 2>/dev/null || true

    echo "  转换完成: ${HF_DIR}"
else
    echo "跳过转换步骤 (SKIP_CONVERT=1)。"
fi

# =============================================================================
# Step 2: lm-eval-harness 评测
# =============================================================================
echo ""
echo "========== Step 2: lm-eval-harness 评测 (5 任务) =========="
echo "任务: ${TASKS}"
echo ""

RESULTS_FILE="${OUT_ROOT}/results_baseline.txt"
echo "========== bf16 baseline 下游任务评测结果 ==========" > "$RESULTS_FILE"
echo "任务: ${TASKS}" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

if [ ! -d "$HF_DIR" ] || [ -z "$(ls "$HF_DIR"/*.safetensors 2>/dev/null)" ]; then
    echo "[错误] HF 模型不存在或无权重: ${HF_DIR}"
    exit 1
fi

MODEL_LOG="${LOG_DIR}/lmeval_${MODEL_NAME}.log"
$PYTHON -m lm_eval \
    --model hf \
    --model_args pretrained="${HF_DIR}",trust_remote_code=True,dtype=bfloat16,device=npu:0 \
    --tasks "${TASKS}" \
    --batch_size auto \
    --output_path "${OUT_ROOT}/lmeval_${MODEL_NAME}" \
    --log_samples \
    2>&1 | tee "$MODEL_LOG"

echo "" >> "$RESULTS_FILE"
echo "[bf16 baseline] 结果:" >> "$RESULTS_FILE"
grep -E "acc(_none)?|acc_norm" "$MODEL_LOG" | tail -20 >> "$RESULTS_FILE" 2>/dev/null || true

# =============================================================================
# Step 3: 汇总
# =============================================================================
echo ""
echo "========== 评测完成 =========="
echo "结果汇总: ${RESULTS_FILE}"
echo ""
cat "$RESULTS_FILE"
