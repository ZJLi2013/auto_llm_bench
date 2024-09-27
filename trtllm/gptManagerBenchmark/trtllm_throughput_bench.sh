#!/bin/bash

MODEL_ROOT="/workspace/models/"

mk_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then    
        echo "create new folder '$dir' "
        mkdir -p "$dir"
    fi 
}

clean_dir() {
    local dir="$1"
    if [ "$(ls -A "$dir")" ]; then
        echo " Clean up folder '$dir' "
        rm -rf "$dir"
    fi 
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --requests) REQUESTS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --prompt_len) PROMPT_LEN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --new_tokens) NEW_TOKENS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --data_type) DATA_TYPE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in 
        --tp) TP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;; 
    esac 
    shift 
done

REQUESTS_SP=""
for i in $(echo $REQUESTS | tr "," "\n")
do
  REQUESTS_SP="$REQUESTS_SP $i"
done

PROMPT_LEN_SP=""
for i in $(echo $PROMPT_LEN | tr "," "\n")
do
  PROMPT_LEN_SP="$PROMPT_LEN_SP $i"
done

NEW_TOKENS_SP=""
for i in $(echo $NEW_TOKENS | tr "," "\n")
do
  NEW_TOKENS_SP="$NEW_TOKENS_SP $i"
done

DATA_TYPE_SP=""
for i in $(echo $DATA_TYPE | tr "," "\n")
do
  DATA_TYPE_SP="$DATA_TYPE_SP $i"
done

if [ -z "$MODEL_NAME" ]; then
    echo "Error: Missing one or more required parameters."
    usage
fi

echo "=hyper params start="
echo $MODEL_NAME
echo $REQUESTS_SP
echo $PROMPT_LEN_SP
echo $NEW_TOKENS_SP
echo $DATA_TYPE_SP
echo $TP


hf_model_path=$MODEL_ROOT/$MODEL_NAME
trtllm_ckpt_path=$MODEL_ROOT/trtllm_ckpt/$MODEL_NAME 
engine_dir=$MODEL_ROOT/engines/$MODEL_NAME 
tp_size=$TP
data_type=$DATA_TYPE_SP
requests=$REQUESTS_SP
isl=$PROMPT_LEN_SP
osl=$NEW_TOKENS_SP

echo "=hyper params end="

# dataset preparation
for i in $isl; do
    for o in $osl; do
        echo "dataset preparation"
        echo "$i, $o"
        dataset_file="/workspace/dataset/${MODEL_NAME}_${i}_${o}"
        dataset_dir=$(dirname "$dataset_file")
        mk_dir $dataset_dir 
        python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py --output=$dataset_file  --tokenizer=$hf_model_path token-norm-dist \
            --num-requests $requests \
            --input-mean=$i \
            --output-mean=$o \
            --input-stdev=0 \
            --output-stdev=0
    done
done

# trtlllm engine build 
for dt in $data_type; do
    if [[ "$MODEL_NAME" == "Meta-Llama-3"* ]]; then
        if ls "$trtllm_ckpt_path"/*.safetensors 1> /dev/null 2>&1; then 
            echo "convert done previously"
        else  
            # weights convert: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama
            python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py  --model_dir $hf_model_path --dtype $data_type --output_dir $trtllm_ckpt_path --tp_size $tp_size 
            if [ $? -ne 0 ]; then 
                echo " '$MODEL_NAME' ck_convert failed, EXIT" 
                exit 1 
            else 
                echo " '$MODEL_NAME' ck_convert done"
            fi   
        fi       
        max_num_tokens=4096
        max_batch_size=2048
    elif [[ "$MODEL_NAME" == "Mixtral-8x22B-v0.1" ]]; then
        if ls "$trtllm_ckpt_path"/*.safetensors 1> /dev/null 2>&1; then 
            echo "convert done previously"
        else      
            # weights convert: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mixtral
            python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py  --model_dir $hf_model_path --dtype $data_type --output_dir $trtllm_ckpt_path --tp_size $tp_size --moe_tp_size 2 --moe_ep_size 4 
            if [ $? -ne 0 ]; then 
                echo " '$MODEL_NAME' ck_convert failed, EXIT" 
                exit 1 
            else 
                echo " '$MODEL_NAME' ck_convert done"
            fi 
        fi 
        max_num_tokens=8192
        max_batch_size=1024
    else 
        echo "TODO"
    fi 

    if ls "$engine_dir"/*.engine 1> /dev/null 2>&1; then 
        echo "using existed engine: $engine_dir for benchmark" 
    else
        echo "building $engine_dir for benchmark"
        # LLama3.1 exceptions
        if [[ "$MODEL_NAME" == "Meta-Llama-3.1"* ]]; then
            trtllm-build --checkpoint_dir $trtllm_ckpt_path \
                --use_fused_mlp \
                --gpt_attention_plugin $data_type \
                --output_dir $engine_dir \
                --workers $tp_size \
                --max_num_tokens 4096 \
                --max_input_len 64000 \
                --max_seq_len 65000 \
                --use_paged_context_fmha enable 
        elif [[ "$MODEL_NAME" == "Meta-Llama-3"* ]]; then
            trtllm-build --checkpoint_dir $trtllm_ckpt_path \
                --use_fused_mlp \
                --gemm_plugin $data_type \
                 --gpt_attention_plugin $data_type \
                --output_dir $engine_dir \
                --workers $tp_size \
                --max_num_tokens 4096 \
                --max_seq_len 65000  \
                --use_paged_context_fmha enable             
        elif (! [ -d ../$engine_dir ] || [ $engine_rebuild = 1 ] ); then
            trtllm-build --checkpoint_dir $trtllm_ckpt_path \
                --use_fused_mlp \
                --gpt_attention_plugin $data_type \
                --output_dir $engine_dir \
                --workers $tp_size \
                --max_batch_size $max_batch_size \
                --max_input_len 8192 \
                --max_num_tokens $max_num_tokens \
                --reduce_fusion disable \
                --use_paged_context_fmha enable \
                --builder_opt=4 \
                --multiple_profiles enable 
        fi
    fi 
done

# trtlllm run
final_report="/workspace/logs/perf_$MODEL_NAME.csv"
final_report_dir=$(dirname "$final_report")
mode="throughput"
mk_dir $final_report_dir 
for dt in $data_type; do
    for o in $osl; do
        for i in $isl; do
            echo "trtllm run"
            echo "$dt, $i, $o"
            dataset_file="/workspace/dataset/${MODEL_NAME}_${i}_${o}"
            results_csv="results_${MODEL_NAME}_${dt}_${i}_${o}"
            mpirun -n $tp_size --allow-run-as-root --oversubscribe /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir $engine_dir --type IFB --dataset $dataset_file  --eos_id -1 --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.99 --output_csv $results_csv --request_rate -1.0 --enable_chunked_context --streaming --warm_up 0
            if [ $? -eq 0 ]; then 
                echo "benchmark running done well, generating reports"
                python generate_reports.py --model_name $MODEL_NAME --mode $mode --tp_size $TP --num_requests $requests --isl $i --osl $o --dtype $dt --trtllm_rpt $results_csv --dlm_csv $final_report
                echo ""
            fi 
        done
    done
done