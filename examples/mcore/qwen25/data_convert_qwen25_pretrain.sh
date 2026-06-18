# 请按照实际环境修改 set_env.sh 路径
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH=/home/MindSpeed:/home/MindSpeed-LLM:$PYTHONPATH

 # 注意路径是否一致,预训练数据集会生成alpaca_text_document.bin和.idx
python3.10 ./preprocess_data.py \
  --input ../dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ../model_from_hf/qwen2.5-7b-hf/ \
  --output-prefix ../dataset/alpaca \
  --tokenizer-type PretrainedFromHF \
  --workers 4 \
  --log-interval 1000