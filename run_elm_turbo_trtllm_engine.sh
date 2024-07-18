#!/bin/bash
set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <elm-turbo_model_id> <gpu_type> '<input_prompt>'"
    echo "Example command to run A100 engine: $0 slicexai/elm-turbo-0.50-instruct A100 'plan a fun day with my grandparents.'"
    echo "Example command to run H100 engine: $0 slicexai/elm-turbo-0.50-instruct H100 'plan a fun day with my grandparents.'"
    echo "Supported 'elm-turbo_model_id' choices : [slicexai/elm-turbo-0.50-instruct, slicexai/elm-turbo-0.25-instruct, slicexai/elm-turbo-0.125-instruct]"
    echo "Supported gpt_types : [A100, H100]"
    exit 1
fi

ELM_TURBO_HF_MODEL_DIR=$1
GPU_TYPE=$2
PROMPT=$3

echo "ELM_TURBO_HF_MODEL_DIR:"$ELM_TURBO_HF_MODEL_DIR
echo "GPU_TYPE:"$GPU_TYPE
echo "PROMPT:"$PROMPT

ENGINE_DIR="${ELM_TURBO_HF_MODEL_DIR}-trtllm-${GPU_TYPE}"

cd /lm/TensorRT-LLM/examples
huggingface-cli download ${ENGINE_DIR} --local-dir ${ENGINE_DIR}
huggingface-cli download ${ELM_TURBO_HF_MODEL_DIR} --local-dir ${ELM_TURBO_HF_MODEL_DIR}

python3 run.py \
  --engine_dir ${ENGINE_DIR} \
  --max_output_len 512 \
  --presence_penalty 0.7 \
  --frequency_penalty 0.7 \
  --temperature 0.0 \
  --tokenizer_dir ${ELM_TURBO_HF_MODEL_DIR} \
  --input_text """<s><|user|>
  ${PROMPT}<|end|>
<|assistant|>
"""
