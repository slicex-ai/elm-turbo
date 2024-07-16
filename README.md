# SliceX AI™ ELM (Efficient Language Models) - Version 0.2
**ELM** (which stands for **E**fficient **L**anguage **M**odels) is the first version in the series of cutting-edge language models from [SliceX AI](https://slicex.ai) that is designed to achieve the best in class performance in terms of _quality_, _throughput_ & _memory_.

<div align="center">
  <img src="elm-rambutan.png" width="256"/>
</div>

ELM is designed to be a modular and customizable family of neural networks that are highly efficient and performant. Today we are sharing the second version in this series: **ELM2** models (named _Rambutan_). 

_Model:_ ELM introduces a new type of _(de)-composable LLM model architecture_ along with the algorithmic optimizations required to learn (training) and run (inference) these models. At a high level, we train a single ELM model in a self-supervised manner (during pre-training phase) but once trained the ELM model can be sliced in many ways to fit different user/task needs. The optimizations can be applied to the model either during the pre-training and/or fine-tuning stage. 

_Fast Inference with Customization:_ Once trained, the ELM model architecture permits flexible inference strategies at runtime depending on the deployment needs. For instance, the ELM model can  be _decomposed_ into smaller slices, i.e., smaller (or larger) models can be extracted from the original model to create multiple inference endpoints. Alternatively, the original (single) ELM model can be loaded _as is_ for inference and different slices within the model can be queried directly to power faster inference. This provides an additional level of flexibility for users to make compute/memory tradeoffs depending on their application and runtime needs.

- **Blog:** [Medium](https://medium.com/sujith-ravi/introducing-elm-efficient-customizable-privacy-preserving-llms-cea56e4f727d)

- **Github:** https://github.com/slicex-ai/elm

- **Demo** (try it out): https://huggingface.co/spaces/slicexai/elm-demo-v1

- **HuggingFace** (access ELM Model cards, code & app from HF): https://huggingface.co/slicexai

## ELM2 Model Release
In our second version, we applied our decompossible ELM techniques on a popular open-source LLM - `microsoft/Phi-3-mini-128k-instruct` (TODO - Refer to phi3-licence). Post training, we generate four slices of varying sizes ranging from 1.33B - 2.91B params. Additionally, we integrated these slices into NVIDIA's [trtllm](https://github.com/NVIDIA/TensorRT-LLM) and present you the trtllm engines compatible for A100 and H100 GPUs resepctively.

## 1. Run ELM2 models with Huggingface Transformers library.
There are three slices derived from the `phi3-mini` (3.82B params) model - 1. `slicexai/elm2-0.125-instruct` (1.33B params), 2. `slicexai/elm2-0.25-instruct`(1.56B params), 3. `slicexai/elm2-0.50-instruct` (2.01B params). 

Required packages for [Hugginface Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct).
```bash
flash_attn==2.5.8
torch==2.3.1
accelerate==0.31.0
transformers==4.41.2
```

Example - To run the `slicexai/elm2-0.50-instruct`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

elm2_model = "slicexai/elm2-0.50-instruct"
model = AutoModelForCausalLM.from_pretrained( 
    elm2_model,  
    device_map="cuda",  
    torch_dtype=torch.bfloat16,  
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
messages = [ 
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
]

tokenizer = AutoTokenizer.from_pretrained(elm2_model, legacy=False) 
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

## 2. Running ELM2 via Nvidia's TensorRT-LLM
You can either use the instructions provided by Nvidia in [trtllm](<link>). To run on edge, (windows RTX), please refer to the instructions at https://github.com/NVIDIA/TensorRT-LLM/blob/main/windows/README.md.

or We have pre-built ELM2-trtllm engines for A100 and H100 GPUS. Below are the instuctions to install and run them.

### a. Download & install Nvidia's TensorRT-LLM with docker.
```
git clone https://github.com/slicex-ai/elm2.git
cd elm2
sh setup_trtllm.sh
```
This creates a docker named `elm_trtllm` and installs tensorrt_llm.

### b. Run pre-built ELM2-trtllm engines with your input prompts.

Example: To run our pre-built trt-engine for `slicexai/elm2-0.50-instruct` on a A100 & H100 gpus respectively,
```
docker attach elm_trtllm
cd /lm
sh run_elm2_trtllm_engine.sh slicexai/elm2-0.50-instruct A100 "plan a fun day with my grandparents."
sh run_elm2_trtllm_engine.sh slicexai/elm2-0.50-instruct H100 "plan a fun day with my grandparents."
```

Detailed instructions to run the engine:
```
Usage: sh run_elm2_trtllm_engine.sh <elm2_model_id> <gpu_type> "<input_prompt>"
Supported elm2_model_id choices : [slicexai/elm2-0.50-instruct, slicexai/elm2-0.25-instruct, slicexai/elm2-0.125-instruct]
Supported gpu_types : [A100, H100]
```


### c. (Optional) Create & run your own ELM2-trtllm engines from ELM2 Huggingface(HF) checkpoints.

#### Compile the Model into a TensorRT Engine

```bash
docker attach elm_trtllm
cd /lm/TensorRT-LLM/examples/phi
pip install -r requirements.txt
python3 convert_checkpoint.py --model_dir <hf_elm2_checkpoint> --output_dir <trtllm_slice_checkpoint>
trtllm-build --checkpoint_dir <trtllm_slice_checkpoint> \
    --gemm_plugin bfloat16 \
    --output_dir <trtllm_slice_engine>
```

#### Run the Model
Now that you’ve got your model engine, its time to run it.

```bash
python3 ../run.py \
  --engine_dir <trtllm_slice_engine> \
  --max_output_len 100 \
  --tokenizer_dir <hf_elm2_checkpoint> \
  --input_text """<s><|user|>
plan a fun day with my grandparents.<|end|>
<|assistant|>
"""
```
