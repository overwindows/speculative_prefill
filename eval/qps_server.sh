# bash eval/qps_server.sh
# SPEC_CONFIG_PATH=./configs/config_p3_full.yaml ENABLE_SP="meta-llama/Meta-Llama-3.1-8B-Instruct" bash eval/qps_server.sh

export SPEC_CONFIG_PATH=./configs/config_p3_full_lah8.yaml
export ENABLE_SP=/nvmedata/hf_checkpoints/Llama-3.2-1B-Instruct

SIZE=${1:-"70B"}
# API_KEY=${2:-local_server}
PORT=${3:-8000}

if [ $SIZE == "70B" ]; then
    MODEL_NAME=/nvmedata/hf_checkpoints/Llama-3.3-70B-Instruct
elif [ $SIZE == "405B" ]; then
    echo "When using 405B model, it is recommended to run on 8xH200."
    MODEL_NAME=neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8
else
    echo "Invalid model size"
    exit 1
fi

# echo "Starting vllm server using port ${PORT} and api key ${API_KEY}"
echo "Starting vllm server using port ${PORT}"

fuser -n tcp ${PORT}

echo ${PORT}

vllm serve /nvmedata/hf_checkpoints/Qwen3-32B --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --max_model_len 131072


# python3 -m speculative_prefill.scripts serve \
#     ${MODEL_NAME} \
#     --tokenizer /nvmedata/hf_checkpoints/Meta-Llama-3.1-8B-Instruct \
#     --dtype auto \
#     --max-model-len 131072 \
#     --gpu-memory-utilization 0.95 \
#     --enable-chunked-prefill=False \
#     --tensor-parallel-size 4 \
#     --max-num-seqs 64 \
#     --port=$PORT

# --enforce-eager \
# --api-key=$API_KEY \
