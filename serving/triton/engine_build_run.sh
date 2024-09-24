#!/bin/bash

# MODEL_NAME=Mixtral-8x22B-v0.1
# requests=128 
# isl=16384
# osl=200
# data_type=float16
# tp_size=4 

usage() {
        echo "Usage: $0 [options ...]"
        echo "Options: "
        echo "  --model_name      Model name to be used (default: Mixtral-8x22B-v0.1)"
        echo "  --requests        Number of requests (default: 128)"
        echo "  --isl             Input sequence length (default: 16384)"
        echo "  --osl             Output sequence length (default: 200)"
        echo "  --data_type       Data type (default: float16)"
        echo "  --tp_size         Tensor parallel size (default: 4)"
        exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model_name) MODEL_NAME="$2"; shift ;;
    --requests) requests="$2"; shift ;;
    --isl) isl="$2"; shift ;;
    --osl) osl="$2"; shift ;;
    --data_type) data_type="$2"; shift ;;
    --tp_size) tp_size="$2"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

max_batch_size=$requests 
max_seq_len=$(( $isl + $osl ))    # [W] padding removal and fMHA are both enabled, max_input_len is not required and will be ignored
max_num_tokens=$(( $max_batch_size * $max_seq_len))  # [W] Specifying a `max_num_tokens` larger than 16384 is usually not recommended, we do not expect perf gain with that and too large `max_num_tokens` could possibly exceed the TensorRT tensor volume, causing runtime errors. Got `max_num_tokens` = 64800

hf_model_path="/workspace/models/${MODEL_NAME}"
trtllm_ckpt_path="/workspace/models/trtllm_ckpt/${MODEL_NAME}"
engine_dir="/workspace/models/engines/${MODEL_NAME}"
dataset_file="/workspace/models/dataset/${MODEL_NAME}_16k_200"

# prepare dataset 
python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py --output=$dataset_file  --tokenizer=$hf_model_path token-norm-dist \
    --num-requests $requests \
    --input-mean=16000 \
    --output-mean=200 \
    --input-stdev=0 \
    --output-stdev=0

if [[ $? -eq 0 ]]; then
        echo "dataset prepared done"
fi

# trtllm_ckpt convert
python3 /app/tensorrt_llm/examples/llama/.py  --model_dir $hf_model_path --dtype $data_type --output_dir $trtllm_ckpt_path --tp_size $tp_size --moe_tp_size 2 --moe_ep_size 2 
if [[ $? -eq 0 ]]; then
        echo "ckpt convert done"
fi

echo $max_batch_size $max_num_tokens $tp_size 

# build trtllm engine for mixtral-8x22b with tp=4 
trtllm-build --checkpoint_dir $trtllm_ckpt_path \
    --gemm_plugin $data_type \
    --output_dir $engine_dir \
    --workers $tp_size \
    --use_paged_context_fmha enable \
    --max_num_tokens $max_num_tokens 
#     --max_batch_size $max_batch_size
#     --builder_opt=4 \
#     --multiple_profiles enable

if [[ $? -eq 0 ]]; then
        echo "engine build done"
fi

mpirun -n $tp_size --allow-run-as-root --oversubscribe /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir $engine_dir --type IFB --dataset $dataset_file  --eos_id -1 --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.9 --request_rate -1.0 --enable_chunked_context --streaming --warm_up 2