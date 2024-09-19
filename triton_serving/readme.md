

## step-by-step to serving Trtllm engine in Triton 


here is a sample to use nvidia triton server to deploy an tensorrt-llm engine

1. [build trtllm engine](./engine_build.sh)
2. following config from [trtllm-backend model repository sample](https://github.com/triton-inference-server/tensorrtllm_backend)
3. launch triton server in terminal
4. launch [perf_bench](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/tools/inflight_batcher_llm/benchmark_core_model.py) in another terminal 


* ngc triton server 24.08 :  trtllm 0.12.0 ,  triton-cli 0.0.11 


```sh
# 1. engine built done with trtllm image :  /workspace/models/engine/Mixtral-8x22B-v0.1 
# 2. prepare model repository
mkdir /triton_model_repo
cp -r /tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_model_repo/
# 3. modify model repositorry 
docker run --gpus '"device=4,5,6,7"' --rm -it --network=host  --shm-size 32g  -v /home/david/nvtriton:/workspace/ -v /data/models/:/workspace/models/  -w /workspace  nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3 
# # generate model repo only once for all 
# cd bytedance/triton_scripts/
# bash generate_model_repo.sh 
# 4. launch serving 
# 'world_size' is the number of GPUs you want to use for serving. This should be aligned with the number of GPUs used to build the TensorRT-LLM engine.
python3 /workspace/tensorrtllm_backend/scripts/launch_triton_server.py --world_size=4 --grpc_port 8101  --http_port 8100 --metrics_port 8102 --model_repo=/workspace/models/triton_model_repo/

# verify by sending an inference request for ensemble_mode 
curl -v localhost:8100/v2/health/ready
curl -X POST localhost:8100/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'

# benchmark run on another terminal 
docker run --gpus '"device=0,1,2,3"' --rm -it --net host -v /home/david/nvtriton:/workspace -v /data/models/:/workspace/models/  nvcr.io/nvidia/tritonserver:24.08-py3-sdk

cd /workspace/tensorrtllm_backend 
pip install -r requirements.txt 

# with dummy input data 
python3 tools/inflight_batcher_llm/benchmark_core_model.py -u localhost:8101 -i grpc  --tensorrt-llm-model-name tensorrt_llm --num-requests 18  --max-input-len 16000 token-norm-dist --input-mean 16000 --input-stdev 0 --output-mean 200 --output-stdev 0
``` 


## Model_Repository Parameters 

WIP

## Triton Perf Analyzer 

WIP 

