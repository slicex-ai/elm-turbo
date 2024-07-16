#!/bin/bash

# This script is based on the installation instructions from:
# https://nvidia.github.io/TensorRT-LLM/installation/linux.html#installing-on-linux

rm -rf TensorRT-LLM
git clone https://github.com/NVIDIA/TensorRT-LLM.git

CURR_FOLDER=$(pwd)
DOCKER_NAME=elm_trtllm
DOCKER_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04

docker run -d --gpus all -it --shm-size=8g --name ${DOCKER_NAME} --ulimit memlock=-1 --rm -v ${CURR_FOLDER}:/lm $DOCKER_IMAGE

command="cd /lm && apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev"
docker exec ${DOCKER_NAME} sh -c "${command}"

command="pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com"
docker exec ${DOCKER_NAME} sh -c "${command}"

command="python3 -c 'import tensorrt_llm'"
docker exec ${DOCKER_NAME} sh -c "${command}"
