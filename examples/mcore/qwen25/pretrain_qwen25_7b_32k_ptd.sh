#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=/home/zs/MindSpeed:/home/zs/MindSpeed-LLM:$PYTHONPATH
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HCCL_CONNECT_TIMEOUT=7200

# 指定使用 2,3,4,5,14,15 这 6 张 NPU 卡
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,14,15

NPUS_PER_NODE=6
MASTER_ADDR=localhost
MASTER_PORT=6010
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
TRAIN_ITERS=300


CKPT_LOAD_DIR="/home/zs/model_weights/qwen2.5_mcore/"
CKPT_SAVE_DIR="/home/zs/ckpt/qwen25-7b"
DATA_PATH="/home/zs/dataset/alpaca_text_document"
TOKENIZER_PATH="/home/zs/model_from_hf/qwen2.5-7b-hf/"

TP=2
PP=1
SEQ_LEN=4096
MBS=1
GBS=66

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --max-position-embeddings ${SEQ_LEN} \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
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
    --metis \
    --metis-quant-dtype fp4 \
    --metis-rank-frac 0.015 \
    --metis-block-size 16 \
    --metis-sample-ratio 0.01 \
    --metis-update-freq 1000 \
    --metis-log-freq 100 \
    --metis-output-dir /home/zs/metis/output
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

CKPT_ARGS="
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    --save ${CKPT_SAVE_DIR}
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 20 \
    --eval-interval 20 \
    --eval-iters 0 \
    --log-throughput \
    --tensorboard-dir /home/zs/MindSpeed-LLM/examples/mcore/qwen25/logs/tensorboard
"

mkdir -p /home/zs/MindSpeed-LLM/examples/mcore/qwen25/logs
mkdir -p /home/zs/MindSpeed-LLM/examples/mcore/qwen25/logs/activation_probe

export ACTIVATION_PROBE_ENABLE=1
export ACTIVATION_PROBE_DIR=/home/zs/MindSpeed-LLM/examples/mcore/qwen25/logs/activation_probe
export ACTIVATION_PROBE_FIRST_N=2
export ACTIVATION_PROBE_LAST_N=2
export ACTIVATION_PROBE_ITERS=4700,4701,4702,4703
export ACTIVATION_PROBE_SAVE_VALUES=1
export ACTIVATION_PROBE_SAMPLE_SIZE=200000
export ACTIVATION_PROBE_BINS=201
export ACTIVATION_PROBE_THRESHOLDS=6,8,10,20,50,100



echo "START_TIME: $(date '+%F %T')"

# 切换到仓库根目录，确保 pretrain_gpt.py 可被找到
cd /home/zs/MindSpeed-LLM

python3.10 -m torch.distributed.run $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    2>&1 | tee examples/mcore/qwen25/logs/pretrain_mcore_qwen25_7b_outliers.log

exit_code=${PIPESTATUS[0]}

