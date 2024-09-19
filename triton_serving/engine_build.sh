MODEL_NAME=Mixtral-8x22B-v0.1
hf_model_path="/workspace/models/${MODEL_NAME}"
# hf_model_name="mistralai/Mixtral-8x22b-Instruct-v0.1"
trtllm_ckpt_path="/workspace/models/trtllm_ckpt/${MODEL_NAME}"
engine_dir="/workspace/models/engines/${MODEL_NAME}"
dataset_file="/workspace/models/dataset/${MODEL_NAME}_16k_200"


requests=128 
max_batch_size=4
max_seq_len=16200 # [W] padding removal and fMHA are both enabled, max_input_len is not required and will be ignored
max_num_tokens=$(( $max_batch_size * $max_seq_len))  # [W] Specifying a `max_num_tokens` larger than 16384 is usually not recommended, we do not expect perf gain with that and too large `max_num_tokens` could possibly exceed the TensorRT tensor volume, causing runtime errors. Got `max_num_tokens` = 64800

data_type=float16
tp_size=4 

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

# it's better to verify the built engine works with IFB 
mpirun -n $tp_size --allow-run-as-root --oversubscribe /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir $engine_dir --type IFB --dataset $dataset_file  --eos_id -1 --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.9 --request_rate -1.0 --enable_chunked_context --streaming --warm_up 0