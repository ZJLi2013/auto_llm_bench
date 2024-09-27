VLLM_CSV="$MAD_MODEL_NAME.csv"
#python trtllm_rpt_to_dlm_csv.py --model_name NousResearch/Meta-Llama-3-70B --tp_size 8 --isl 128 --osl 128 --trtllm_rpt results_fp8_128_128 --dlm_csv omg.csv
import csv, sys
import argparse
import os
import datetime  

# parse arg
parser = argparse.ArgumentParser(description='Convert vllm csv output format to DLM csv output format')
parser.add_argument("--model_name",
                        type=str,
                        help="model name")
parser.add_argument("--mode",   
                        type=str,
                        help="reports mode: throughput | latency")
parser.add_argument("--metrics",
                        metavar="S",
                        type=str,
                        nargs="+",
                        default= "throughput",
                        help="reports metrics: throughput, latency")
parser.add_argument("--tp_size",
                        type=str,
                        help="tensorparallel size")
parser.add_argument("--num_requests",
                        type=str,
                        help="input requests")
parser.add_argument("--isl",
                        type=str,
                        help="input seq length")
parser.add_argument("--osl",
                        type=str,
                        help="output seq length")
parser.add_argument("--dtype",
                        type=str,
                        help="data_type")
parser.add_argument("--trtllm_rpt",
                        help="path to the trtllm_rpt file")
parser.add_argument("--dlm_csv",
                        help="path to the dlm_csv file")

# read args
args = parser.parse_args()
line_idx = 4
############### sample output from GPTManagerBenchmark ###################
# [BENCHMARK] num_error_samples 0

# [BENCHMARK] num_samples 2000
# [BENCHMARK] total_latency(ms) 265197.47
# [BENCHMARK] seq_throughput(seq/sec) 7.54
# [BENCHMARK] token_throughput(token/sec) 7.54    # line 4

# [BENCHMARK] avg_sequence_latency(ms) 530.37     # line 5
# [BENCHMARK] max_sequence_latency(ms) 771.90
# [BENCHMARK] min_sequence_latency(ms) 513.99
# [BENCHMARK] p99_sequence_latency(ms) 534.03
# [BENCHMARK] p90_sequence_latency(ms) 532.29
# [BENCHMARK] p50_sequence_latency(ms) 530.07

# [BENCHMARK] avg_time_to_first_token(ms)        # line 11 
# [BENCHMARK] max_time_to_first_token(ms)
# [BENCHMARK] min_time_to_first_token(ms)
# [BENCHMARK] p99_time_to_first_token(ms)
# [BENCHMARK] p90_time_to_first_token(ms)
# [BENCHMARK] p50_time_to_first_token(ms)

# [BENCHMARK] avg_inter_token_token(ms)          # line 17
# [BENCHMARK] mavg_inter_token_token(ms)
# [BENCHMARK] avg_inter_token_token(ms)
# [BENCHMARK] p99_avg_inter_token(ms)
# [BENCHMARK] p90_avg_inter_token(ms)
# [BENCHMARK] p50_avg_inter_token(ms)


if args.mode == "throughput":
    line_idx = 4
elif args.mode == "latency":
    line_idx = 5
else :
    print("un-supported runtime mode")
    exit(-1)

with open(args.trtllm_rpt, newline='') as inpf:
    header_write = 0 if os.path.exists(args.dlm_csv) else 1
    with open(args.dlm_csv,'a+',newline='') as outf:
        writer = csv.writer(outf, delimiter=',')
        if header_write:
            writer.writerow(['model', 'performance', 'metric']) if header_write else None
        reader = csv.reader(inpf)
        writer.writerow([datetime.datetime.now()])
        try:
            for row in reader:
                if(args.mode == "throughput" and row[line_idx] == "token_throughput(token/sec)"):
                    metric_name = " THROUGHPUT_GEN"
                    continue
                elif (args.mode == "latency" and row[line_idx] == "avg_sequence_latency(ms)"):
                    metric_name = "LATENCY_STATIC_BS"
                    continue 
                else:
                    metric = 'token/sec'
                    model_details = args.model_name + metric_name + ' tp' + args.tp_size + ' requests' + args.num_requests + ' in' + args.isl + ' out' + args.osl  + ' ' + args.dtype
                    writer.writerow([model_details, " " + row[line_idx], " " + metric])

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(args.trtllm_rpt, reader.line_num, e))
