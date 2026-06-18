# 请按照实际环境修改 set_env.sh 路径
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH=/home/MindSpeed:/home/MindSpeed-LLM:$PYTHONPATH
# 将切分调整为tp1,将切分调整为pp4
python3.10 convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 4 \
       --add-qkv-bias \
       --load-dir ../model_from_hf/qwen2.5-7b-hf/ \
       --save-dir ../model_weights/qwen2.5_mcore/ \
       --tokenizer-model ../model_from_hf/qwen2.5-7b-hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16