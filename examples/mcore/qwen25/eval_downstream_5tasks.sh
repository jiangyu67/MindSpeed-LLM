#!/bin/bash
# =============================================================================
# 下游任务评测脚本：3 个模型 × 5 个任务 (ARC-C, ARC-E, BoolQ, LAMBADA, PIQA)
#
# 流程：
#   1. 将 3 个 Megatron checkpoint 转换为 HF 格式
#   2. 安装 lm-eval-harness（若未安装）
#   3. 对每个 HF 模型跑 5 个下游任务
#   4. 汇总结果
#
# 前提：三个训练（bf16 / metis / averis）已完成，checkpoint 已保存。
#
# 用法：
#   bash /home/zs/MindSpeed-LLM/examples/mcore/qwen25/eval_downstream_5tasks.sh
#
# 可选环境变量：
#   ASCEND_RT_VISIBLE_DEVICES  指定使用的 NPU 卡（默认 2,3,4,5,14,15）
#   SKIP_CONVERT=1             跳过转换步骤（已转换过时使用）
#   SKIP_INSTALL=1             跳过 lm-eval 安装
# =============================================================================

set -euo pipefail

# ---- 环境配置 ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONPATH=/home/zs/MindSpeed:/home/zs/MindSpeed-LLM:$PYTHONPATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}  # HF 镜像，避免网络问题

# NPU 卡设置（lm-eval 单卡推理即可，默认用 1 张卡；可按需修改）
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

PYTHON=/root/miniconda3/envs/qwen/bin/python3.10
PIP="/root/miniconda3/envs/qwen/bin/pip3.10"
REPO_DIR=/home/zs/MindSpeed-LLM
HF_REF_DIR=/home/zs/model_from_hf/qwen2.5-7b-hf          # HF 参考模型（提供 config/tokenizer）
OUT_ROOT=/home/zs/ckpt/eval_downstream
LOG_DIR=${OUT_ROOT}/logs
mkdir -p "$LOG_DIR"

# ---- 三个模型的 checkpoint 配置 ----
# 格式："名称 | Megatron checkpoint 目录 | HF 输出目录"
MODELS=(
    "bf16     | /home/zs/ckpt/qwen25-7b-bf16-baseline    | ${OUT_ROOT}/hf_bf16"
    "metis    | /home/zs/ckpt/qwen25-7b-metis-w4a4g4     | ${OUT_ROOT}/hf_metis"
    "averis   | /home/zs/ckpt/qwen25-7b-mean-bias        | ${OUT_ROOT}/hf_averis"
)

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
    # 读取 latest_checkpointed_iteration.txt，获取最新迭代号
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
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name mg_dir hf_dir <<< "$(echo "$entry" | sed 's/^ *//;s/ *$//;s/ *| */|/g')"
        name=$(echo "$name" | xargs)
        mg_dir=$(echo "$mg_dir" | xargs)
        hf_dir=$(echo "$hf_dir" | xargs)

        echo ""
        echo "---- 转换 [${name}]: ${mg_dir} -> ${hf_dir} ----"

        if [ ! -d "$mg_dir" ]; then
            echo "[警告] checkpoint 目录不存在: ${mg_dir}，跳过 ${name}。"
            continue
        fi

        # 获取最新迭代号
        iter=$(get_latest_iter "$mg_dir")
        if [ -z "$iter" ]; then
            echo "[警告] 未找到 latest_checkpointed_iteration.txt，尝试用 release。"
            iter="release"
        fi
        echo "  使用迭代: ${iter}"

        mkdir -p "$hf_dir"
        # 拷贝 tokenizer / config（转换脚本需要 HF 参考目录）
        cp -n "$HF_REF_DIR"/tokenizer*.json "$hf_dir"/ 2>/dev/null || true
        cp -n "$HF_REF_DIR"/vocab.json "$hf_dir"/ 2>/dev/null || true
        cp -n "$HF_REF_DIR"/merges.txt "$hf_dir"/ 2>/dev/null || true
        cp -n "$HF_REF_DIR"/tokenizer_config.json "$hf_dir"/ 2>/dev/null || true
        cp -n "$HF_REF_DIR"/config.json "$hf_dir"/ 2>/dev/null || true
        cp -n "$HF_REF_DIR"/generation_config.json "$hf_dir"/ 2>/dev/null || true

        cd "$REPO_DIR"
        $PYTHON mindspeed_llm/tasks/checkpoint/convert_param.py \
            --cvt-type mg2hf \
            --model-name llama \
            --mg-dir "$mg_dir" \
            --hf-dir "$hf_dir" \
            --model-config-file "$HF_REF_DIR/config.json" \
            --model-index-file "$HF_REF_DIR/model.safetensors.index.json" \
            --tensor-model-parallel-size $TP \
            --pipeline-model-parallel-size $PP \
            --num-layers $NUM_LAYERS \
            --make-vocab-size-divisible-by 1 \
            --iteration "$iter" \
            2>&1 | tee "${LOG_DIR}/convert_${name}.log"

        echo "  转换完成: ${hf_dir}"
    done
else
    echo "跳过转换步骤 (SKIP_CONVERT=1)。"
fi

# =============================================================================
# Step 2: lm-eval-harness 评测
# =============================================================================
echo ""
echo "========== Step 2: lm-eval-harness 评测 (5 任务 × 3 模型) =========="
echo "任务: ${TASKS}"
echo ""

RESULTS_FILE="${OUT_ROOT}/results_summary.txt"
echo "========== 下游任务评测结果汇总 ==========" > "$RESULTS_FILE"
echo "任务: ${TASKS}" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name mg_dir hf_dir <<< "$(echo "$entry" | sed 's/^ *//;s/ *$//;s/ *| */|/g')"
    name=$(echo "$name" | xargs)
    hf_dir=$(echo "$hf_dir" | xargs)

    echo "---- 评测 [${name}]: ${hf_dir} ----"

    if [ ! -d "$hf_dir" ] || [ -z "$(ls "$hf_dir"/*.safetensors 2>/dev/null)" ]; then
        echo "[警告] HF 模型不存在或无权重: ${hf_dir}，跳过 ${name}。"
        echo "[${name}] 跳过（HF 模型缺失）" >> "$RESULTS_FILE"
        continue
    fi

    MODEL_LOG="${LOG_DIR}/lmeval_${name}.log"
    # lm-eval 单卡推理；torch_npu 会将 cuda 映射到 NPU
    $PYTHON -m lm_eval \
        --model hf \
        --model_args pretrained="${hf_dir}",trust_remote_code=True,dtype=bfloat16 \
        --tasks "${TASKS}" \
        --batch_size auto \
        --output_path "${OUT_ROOT}/lmeval_${name}" \
        --log_samples \
        2>&1 | tee "$MODEL_LOG"

    echo "" >> "$RESULTS_FILE"
    echo "[${name}] 结果:" >> "$RESULTS_FILE"
    # 从日志中提取每个任务的准确率
    grep -E "acc(_none)?|acc_norm" "$MODEL_LOG" | tail -20 >> "$RESULTS_FILE" 2>/dev/null || true

    echo "  ${name} 评测完成，日志: ${MODEL_LOG}"
done

# =============================================================================
# Step 3: 汇总
# =============================================================================
echo ""
echo "========== 评测完成 =========="
echo "结果汇总: ${RESULTS_FILE}"
echo ""
cat "$RESULTS_FILE"
