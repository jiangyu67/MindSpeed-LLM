#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=/home/zs/MindSpeed:/home/zs/MindSpeed-LLM:$PYTHONPATH
export HCCL_CONNECT_TIMEOUT=7200

# 指定使用 2,3,4,5,14,15 这 6 张 NPU 卡
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,14,15

CHECKPOINT="/home/zs/ckpt/qwen25-7b"
TOKENIZER_PATH="/home/zs/model_from_hf/qwen2.5-7b-hf/"
DATA_PATH="/home/zs/ckpt/qwen25-7b/mmlu_data/test"
TASK="mmlu_ppl"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=6
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=2
PP=1
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mkdir -p /home/zs/MindSpeed-LLM/examples/mcore/qwen25/logs

# Different task needs different max_new_tokens value, please follow the instruction in readme.
cd /home/zs/MindSpeed-LLM
python3.10 -m torch.distributed.run $DISTRIBUTED_ARGS evaluation.py \
       --use-mcore-models \
       --task ${TASK} \
       --task-data-path $DATA_PATH \
       --no-chat-template \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 28  \
       --hidden-size 3584  \
       --ffn-hidden-size 18944 \
       --num-attention-heads 28  \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --disable-bias-linear \
       --add-qkv-bias \
       --group-query-attention \
       --num-query-groups 4 \
       --swiglu \
       --use-fused-swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --use-fused-rmsnorm \
       --position-embedding-type rope \
       --rotary-base 1000000 \
       --use-fused-rotary-pos-emb \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --micro-batch-size 1  \
       --max-new-tokens 1 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --untie-embeddings-and-output-weights \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --load ${CHECKPOINT} \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --transformer-impl local \
       | tee /home/zs/MindSpeed-LLM/examples/mcore/qwen25/logs/evaluation_mcore_qwen25_7b_${TASK}.log
