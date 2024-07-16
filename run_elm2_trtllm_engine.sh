

CURR_FOLDER=$(pwd)
DOCKER_NAME=elm_trtllm

if [ $# -ne 3 ]; then
    echo "Usage: $0 <hf_elm2_model_id> <gpu_type> <input_prompt>"
    echo "Example command to run A100 engine: $0 slicexai/elm2-0.50-instruct A100 'Can you provide ways to eat combinations of bananas and dragonfruits?'"
    echo "Example command to run H100 engine: $0 slicexai/elm2-0.50-instruct H100"
    echo "Supported 'elm2_model_id' choices : [elm2-0.50-instruct, elm2-0.25-instruct, elm2-0.125-instruct]"
    echo "Supported GPU types : [A100, H100]"
    exit 1
fi

ELM2_HF_MODEL_DIR=$1
GPU_TYPE=$2
PROMPT=$3

echo "ELM2_HF_MODEL_DIR:"$ELM2_HF_MODEL_DIR
echo "GPU_TYPE:"$GPU_TYPE
echo "PROMPT:"$PROMPT

ENGINE_DIR="${ELM2_HF_MODEL_DIR}-trtllm-${GPU_TYPE}"

cd /lm/TensorRT-LLM/examples
huggingface-cli download slicexai/${ENGINE_DIR} --local-dir ${ENGINE_DIR}
huggingface-cli download slicexai/${ELM2_HF_MODEL_DIR} --local-dir ${ELM2_HF_MODEL_DIR}

python3 run.py \
  --engine_dir ${ENGINE_DIR} \
  --max_output_len 128 \
  --tokenizer_dir ${ELM2_HF_MODEL_DIR} \
  --input_text """<s><|user|>
${PROMPT}<|end|>
<|assistant|>
"""