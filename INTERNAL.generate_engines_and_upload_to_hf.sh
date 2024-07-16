#!/bin/bash
GPU="A100"
cd /lm/TensorRT-LLM/examples/phi
pip install flash_attn
#"slicexai/elm2-0.50-instruct"  "slicexai/elm2-0.25-instruct" "slicexai/elm2-0.125-instruct"
for hf_model_dir in "slicexai/elm2-0.25-instruct" "slicexai/elm2-0.125-instruct"
do
    echo "Building $hf_model_dir"
    huggingface-cli download $hf_model_dir --local-dir ../$hf_model_dir
    python3 convert_checkpoint.py --dtype bfloat16 --use_weight_only --weight_only_precision int8  --model_dir ../$hf_model_dir --output_dir ../$hf_model_dir-trtllm-ckpt
    trtllm-build --gpt_attention_plugin bfloat16 --gemm_plugin bfloat16 --max_seq_len 4096 --max_batch_size 256 --checkpoint_dir ../$hf_model_dir-trtllm-ckpt --output_dir ../$hf_model_dir-trtllm-engine

    huggingface-cli upload $hf_model_dir-trtllm-$GPU ../$hf_model_dir-trtllm-engine
done





