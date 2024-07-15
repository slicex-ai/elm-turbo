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
In our second version, we applied our decompossible ELM techniques on a popular open-source LLM - `microsoft/Phi-3-mini-128k-instruct`. We release four slices of the `Phi-3-mini-128k-instruct` model. Additionally, we integrated these slices into NVIDIA's [trtllm](https://github.com/NVIDIA/TensorRT-LLM) and present you the trtllm engines compatible for A100 and H100 GPUs resepctively.

## Setup ELM
```bash
git clone https://github.com/slicex-ai/elm2
cd elm2
sudo apt-get install git-lfs 
git lfs install
sh download_models.sh
```
