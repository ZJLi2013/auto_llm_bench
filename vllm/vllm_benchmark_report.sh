#!/usr/bin/bash

#Usage: 

while getopts s:m:g:d:q: flag
do
    case "${flag}" in
        s) scenario=${OPTARG};;
        m) model=${OPTARG};;
        g) numgpu=${OPTARG};;
        d) datatype=${OPTARG};;
        q) quantization=${OPTARG};; 
    esac
done
echo "MODEL: $model ";

# only for ROCM
export HIP_FORCE_DEV_KERNARG=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1
export VLLM_TUNE_GEMM=0

# args
model_name=$(basename "$model")
dtype=$datatype
tp=$numgpu
quant=$quantization

# latency conditions
Bat="1 2 4 8 16 32 64 128 256"
# decoding benchmark
InLatency="1"
OutLatency="128"
# # ttft benchmark
# FTL_Bat="4 8 16"
# FTL_In="128 4096 8192 32768"
# FTL_Out="1" 



# throughput conditions
Req="256"
InThroughput="128 2048"
OutThroughput="128 2048"

report_dir="/vllm-workspace/reports_${dtype}"
tool_latency="/app/vllm/benchmarks/benchmark_latency.py"
tool_throughput="/app/vllm/benchmarks/benchmark_throughput.py"
tool_report="vllm_benchmark_report.py"
n_warm=3
n_itr=5
mkdir -p $report_dir

if [ "$scenario" == "latency" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] LATENCY"
    mode="latency"
    for bat in $Bat;
    do
        for out in $OutLatency;
        do        
            for inp in $InLatency;
            do
                outjson=${report_dir}/${model_name}_${mode}_prefill_bs${bat}_in${inp}_out${out}_${dtype}.json
                outcsv=${report_dir}/${model_name}_${mode}_report.csv
                if [ "$inp" -gt 8192 ]; then 
                    max_model_len=$inp 
                else 
                    max_model_len=8192 
                fi  
                echo $model $mode $bat $tp $inp $out $max_model_len $quant 
                if [ -n "$quant" ]; then 
                    echo "using $quant running quant inference"
                    python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype auto --max_model_len $max_model_len --enforce-eager --output-json $outjson --quantization $quant 
                else 
                    if [ $tp -eq 1 ]; then
                        python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype $dtype  --max_model_len $max_model_len --enforce-eager --output-json $outjson
                    else
                        # torchrun for tp>1 has oom issue
                        # torchrun --standalone --nnodes 1 --nproc-per-node $tp $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype $dtype  --max_model_len $max_model_len --output-json $outjson
                        python3 $tool_latency --model $model --batch-size $bat -tp $tp --input-len $inp --output-len $out --num-iters-warmup $n_warm --num-iters $n_itr --trust-remote-code --dtype $dtype  --max_model_len $max_model_len --enforce-eager --output-json $outjson                    
                    fi
                fi 
                if [ $? -eq 0 ]; then 
                    echo "benchmark running done well, generating reports"            
                    python3 $tool_report --mode ${mode} --model $model_name --batch-size $bat --tp $tp --input-len $inp --output-len $out --dtype $dtype --input-json $outjson --output-csv $outcsv
                else 
                    echo "benchmark running failed, please check"
                    continue  
                fi            
            done
        done 
    done
fi

if [ "$scenario" == "throughput" ] || [ "$scenario" == "all" ]; then
    echo "[INFO] THROUGHPUT"
    mode="throughput"
    for req in $Req;
    do
        for out in $OutThroughput;
        do
            for inp in $InThroughput;
            do
                outjson=${report_dir}/${model_name}_${mode}_req${req}_in${inp}_out${out}_${dtype}.json
                outcsv=${report_dir}/${model_name}_${mode}_report.csv
                echo $model $mode $req $tp $inp $out $quant 
                if [ -n "$quant" ] ; then
                    echo "using $quant running quant inference"
                    python3 $tool_throughput --model $model --num-prompts $req -tp $tp --distributed-executor-backend mp --input-len $inp --output-len $out --trust-remote-code --dtype auto --enforce-eager --output-json $outjson   --quantization $quant 
                else 
                    if [ $tp -eq 1 ]; then
                        python3 $tool_throughput --model $model --num-prompts $req -tp $tp --distributed-executor-backend mp --input-len $inp --output-len $out --trust-remote-code --dtype $dtype --enforce-eager --output-json $outjson 
                         
                    else
                        python3 $tool_throughput --model $model --num-prompts $req -tp $tp --distributed-executor-backend mp --input-len $inp --output-len $out --trust-remote-code --dtype $dtype --enforce-eager --output-json $outjson               
                    fi
                fi 
                if [ $? -eq 0 ]; then 
                    echo "benchmark running done well, generating reports"
                    python3 $tool_report --mode $mode --model $model_name --num-prompts $req --tp $tp --input-len $inp --output-len $out --dtype $dtype --input-json $outjson --output-csv $outcsv
                else 
                    echo "benchmark running failed, please check"
                    continue  
                fi 
            done
        done
    done
fi
