for seq_len in 32768 65536
do

for out_len in 150 1
do

one=1

if [ ${out_len} -eq ${one} ] ; then
out_len=$(($seq_len / 4))
fi

for jobs in 16 64
do

export JOBS=$jobs
export TIN=$seq_len
export TOUT=$out_len
export OPENAI_API_KEY=token-deepauto 
export OPENAI_API_BASE="http://localhost:8000/v1/" 

echo ">>>>>>>>>> EXPERIMENT CONFIG <<<<<<<<<< $JOBS $TIN $TOUT"

python token_benchmark_ray.py \
    --model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
    --mean-input-tokens $TIN \
    --stddev-input-tokens 150 \
    --mean-output-tokens $TOUT \
    --stddev-output-tokens 10 \
    --max-num-completed-requests $JOBS \
    --timeout 60000 \
    --num-concurrent-requests $JOBS \
    --results-dir "result_outputs" \
    --llm-api openai \
    --additional-sampling-params '{}'

sleep 10

done

done

done