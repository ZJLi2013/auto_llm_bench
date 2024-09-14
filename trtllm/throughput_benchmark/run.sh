#!/usr/bin/bash
set -ex

# export NCCL_DEBUG=INFO 
export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

bash trtllm_throughput_bench.sh --model_name "Mixtral-8x22B-v0.1" --requests 512  --prompt_len 1024,4096,8192,16384 --new_tokens 200 --data_type bfloat16  --tp 4
