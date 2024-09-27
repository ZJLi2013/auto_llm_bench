

MODEL_ROOT="/workspace/models/"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
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
    case $1 in 
        --static_bs) STATIC_BS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;; 
    esac 
    shift 
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

STATIC_BATCH_SIZE=""
for i in $(echo $STATIC_BS | tr "," "\n")
do 
    STATIC_BATCH_SIZE="$STATIC_BATCH_SIZE $i"
done 

if [ -z "$MODEL_NAME" ]; then
    echo "Error: Missing one or more required parameters."
    usage
fi

echo "=hyper params start="
echo $MODEL_NAME
echo $PROMPT_LEN_SP
echo $NEW_TOKENS_SP
echo $DATA_TYPE_SP
echo $TP
echo $STATIC_BATCH_SIZE 

engine_rebuild=0
hf_model_path=$MODEL_ROOT/$MODEL_NAME
trtllm_ckpt_path=$MODEL_ROOT/trtllm_ckpt/$MODEL_NAME 
engine_dir=$MODEL_ROOT/engines/$MODEL_NAME 
tp_size=$TP
data_type=$DATA_TYPE_SP
isl=$PROMPT_LEN_SP
osl=$NEW_TOKENS_SP
bs=$STATIC_BATCH_SIZE 

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
    elif [[ "$MODEL_NAME" == "Mixtral-8x22B-v0.1" ]]; then
        if ls "$trtllm_ckpt_path"/*.safetensors 1> /dev/null 2>&1; then 
            echo "convert done previously"
        else      
            # weights convert: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mixtral
            ep_size=$(($tp_size/2))
            python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py  --model_dir $hf_model_path --dtype $data_type --output_dir $trtllm_ckpt_path --tp_size $tp_size --moe_tp_size 2 --moe_ep_size $ep_size
            if [ $? -ne 0 ]; then 
                echo " '$MODEL_NAME' ck_convert failed, EXIT" 
                exit 1 
            else 
                echo " '$MODEL_NAME' ck_convert done"
            fi 
        fi 
    else 
        echo "TODO"
    fi 
done

for dt in $data_type; do 
        for batch_size in $bs; do 
                for in_num in $isl; do
                    for out_num in $osl; do 
                        in_out_dims="${in_num},${out_num}"
                        echo "BS: $batch_size, ISL/OSL: $in_out_dims"
                        token_num=$(($batch_size*($in_num+$out_num)))
                        echo "token_num: $token_num"
                        trtllm-build --checkpoint_dir $trtllm_ckpt_path  --output_dir $engine_Dir --gemm_plugin bfloat16 --workers $tp_size --max_num_tokens $token_num --gemm_plugin $data_type --use_fused_mlp  --gpt_attention_plugin $data_type --use_paged_context_fmha enable  
                        mpirun --allow-run-as-root -n $tp_size /app/tensorrt_llm/benchmarks/cpp/gptSessionBenchmark --engine_dir $engine_dir --warm_up 2 --batch_size $batch_size  --num_runs 10 --input_output_len $in_out_dims  >> report_${MODEL_NAME}_b${batch_size}_isl${in_num}_osl${out_num}.log  2>&1
                done 
        done 
done