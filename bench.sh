export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:$SRT_PORT/v1/"

for seq_len in 32000 64000 128000 256000 512000 1024000 2048000 3072000
do
out_len=2048
max_new_token=$out_len
min_new_token=$((max_new_token - 1))

for jobs in 1
do

total_jobs=$((jobs * 4))
echo ">>>>>>>>>> EXPERIMENT CONFIG <<<<<<<<<< input len ${seq_len}, output len ${out_len}, jobs ${total_jobs} (${jobs})"
python -u token_benchmark_ray.py \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--mean-input-tokens $seq_len \
--stddev-input-tokens 0 \
--mean-output-tokens $out_len \
--stddev-output-tokens 0 \
--max-num-completed-requests $total_jobs \
--timeout 60000 \
--num-concurrent-requests $jobs \
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params "{\"ignore_eos\": true, \"min_tokens\": $max_new_token, \"max_new_tokens\": $max_new_token, \"min_new_tokens\": $max_new_token}"

done
done
