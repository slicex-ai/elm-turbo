# SliceX AI™ ELM (Efficient Language Models) - Version 0.2
**ELM** (which stands for **E**fficient **L**anguage **M**odels) is the first version in the series of cutting-edge language models from [SliceX AI](https://slicex.ai) that is designed to achieve the best in class performance in terms of _quality_, _throughput_ & _memory_.

<div align="center">
  <img src="elm-rambutan.png" width="256"/>
</div>

ELM is designed to be a modular and customizable family of neural networks that are highly efficient and performant. Today we are sharing the second version in this series: **ELM-v0.2** models (named _Rambutan_). 

_Model:_ ELM introduces a new type of _(de)-composable LLM model architecture_ along with the algorithmic optimizations required to learn (training) and run (inference) these models. At a high level, we train a single ELM model in a self-supervised manner (during pre-training phase) but once trained the ELM model can be sliced in many ways to fit different user/task needs. The optimizations can be applied to the model either during the pre-training and/or fine-tuning stage. 

_Fast Inference with Customization:_ Once trained, the ELM model architecture permits flexible inference strategies at runtime depending on the deployment needs. For instance, the ELM model can  be _decomposed_ into smaller slices, i.e., smaller (or larger) models can be extracted from the original model to create multiple inference endpoints. Alternatively, the original (single) ELM model can be loaded _as is_ for inference and different slices within the model can be queried directly to power faster inference. This provides an additional level of flexibility for users to make compute/memory tradeoffs depending on their application and runtime needs.

- **Blog:** [Medium](https://medium.com/sujith-ravi/introducing-elm-efficient-customizable-privacy-preserving-llms-cea56e4f727d)

- **Github:** https://github.com/slicex-ai/elm

- **Demo** (try it out): https://huggingface.co/spaces/slicexai/elm-demo-v1

- **HuggingFace** (access ELM Model cards, code & app from HF): https://huggingface.co/slicexai

## ELM-v0.2 Model Release
In our second version, we applied our decompossible ELM techniques on a popular open-source LLM - `microsoft/Phi-3-mini-128k-instruct`. Post training, we generate four slices of varying sizes ranging from 1.33B - 2.91B params. Additionally, we integrated these slices into NVIDIA's [trtllm](https://github.com/NVIDIA/TensorRT-LLM) and present you the trtllm engines compatible for A100 and H100 GPUs resepctively.

## RUN ELM-v0.2 models with Huggingface Transformers library.
There are four slices derived from the `phi3-mini` (3.82B params) model - 1. `slicexai/elm-v0.2-0.125-instruct` (1.33B params), 2. `slicexai/elm-v0.2-0.25-instruct`(1.56B params), 3. `slicexai/elm-v0.2-0.50-instruct` (2.01B params), 4. `slicexai/elm-v0.2-0.75-instruct` (2.91B params). 

Required packages for [Hugginface Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct).
```bash
flash_attn==2.5.8
torch==2.3.1
accelerate==0.31.0
transformers==4.41.2
```

Example - To run the `slicexai/elm-v0.2-0.50-instruct`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

elmv2_model = "slicexai/elm-v0.2-0.50-instruct"
model = AutoModelForCausalLM.from_pretrained( 
    elmv2_model,  
    device_map="cuda",  
    torch_dtype=torch.bfloat16,  
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
messages = [ 
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
]

tokenizer = AutoTokenizer.from_pretrained(elmv2_model, legacy=False) 
pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text']) 
```

## Setup ELM
```bash
git clone https://github.com/slicex-ai/elm2
cd elm2
sudo apt-get install git-lfs 
git lfs install
sh install_trtllm_with_docker.sh
```

## Run ELMv2-trtllm engines

Example - To run trt-engine for `slicexai/elm-v0.2-0.50-instruct` on a A100 & H100 gpus respectively,
```
sh run_engine.sh "slicexai/elm-v0.2-0.50-instruct-trtllm-A100" Can you provide ways to eat combinations of bananas and dragonfruits?
sh run_engine.sh "slicexai/elm-v0.2-0.50-instruct-trtllm-H100" Can you provide ways to eat combinations of bananas and dragonfruits?
```

## (Optional) Create your ELMv2-trtllm engines from ELMv2 Huggingface(HF) checkpoints.
This step invovles first converting ELMv2 HF slices 

```bash
cd examples/phi3
pip install -r requirements.txt
python3 convert_checkpoint.py --model_dir <hf_slice_checkpoint> --output_dir <trtllm_slice_checkpoint>
trtllm-build --checkpoint_dir <trtllm_slice_checkpoint> \
    --gemm_plugin bfloat16 \
    --output_dir <trtllm_slice_engine>
```



```bash
python3 ../run.py \
  --engine_dir <trtllm_slice_engine> \
  --max_output_len 100 \
  --tokenizer_dir meta-llama/Llama-2-7b-chat-hf \
  --input_text """<s><|user|>
How to setup a human base on Mars? Give short answer<|end|>
<|assistant|>
"""
```
