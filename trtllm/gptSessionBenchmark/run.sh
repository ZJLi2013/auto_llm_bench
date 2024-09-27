#!/usr/bin/bash
set -ex

# export NCCL_DEBUG=INFO 
export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

bash trtllm_latency_bench.sh --model_name "Mixtral-8x22B-v0.1"  --prompt_len 1024,8192,16384 --new_tokens 1,200 --data_type float16  --tp 4 --static_bs 4,8,16

