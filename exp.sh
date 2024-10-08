#!/bin/bash

for seq_len in 8192 16384 32768 65536 131072
do

jobs=32

if [ "${seq_len}" == "8192" ]; then
jobs=100
elif [ "${seq_len}" == "16384" ]; then
jobs=100
elif [ "${seq_len}" == "32768" ]; then
jobs=80
elif [ "${seq_len}" == "65536" ]; then
jobs=40
elif [ "${seq_len}" == "131072" ]; then
jobs=20
else
jobs=32
fi

for out_len in 256 512 1024 2048
do

export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:30000/v1/" 

input_var=$(( seq_len / 10 ))
output_var=$(( out_len / 10 ))
total_jobs=$(( jobs * 4 ))

echo ">>>>>>>>>> EXPERIMENT CONFIG <<<<<<<<<< ${seq_len} += ${input_var}, ${out_len} += ${output_var}, ${total_jobs} (${jobs})"

python -u token_benchmark_ray.py \
    --model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
    --mean-input-tokens $seq_len \
    --stddev-input-tokens $input_var \
    --mean-output-tokens $out_len \
    --stddev-output-tokens $output_var \
    --max-num-completed-requests $jobs \
    --timeout 60000 \
    --num-concurrent-requests $total_jobs \
    --results-dir "result_outputs" \
    --llm-api openai \
    --additional-sampling-params '{}'

done

done