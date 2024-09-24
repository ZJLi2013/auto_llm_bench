


bash  ./engine_build_run.sh --model_name Mixtral-8x22B-v0.1 --requests 18 --isl 16384 --osl 200 --data_type float16 --tp_size 8
bash  ./engine_build_run.sh --model_name Mixtral-8x22B-v0.1 --requests 9 --isl 16384 --osl 200 --data_type float16 --tp_size 4
bash  ./engine_build_run.sh --model_name Mixtral-8x22B-v0.1 --requests 36 --isl 8192 --osl 200 --data_type float16 --tp_size 8
bash  ./engine_build_run.sh --model_name Mixtral-8x22B-v0.1 --requests 18 --isl 8192 --osl 200 --data_type float16 --tp_size 4