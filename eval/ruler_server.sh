# Append ENABLE_SP and SPEC_CONFIG_PATH if nececssary
# export SPEC_CONFIG_PATH=../configs/config_p1_full_lah8.yaml
# export SPEC_CONFIG_PATH=../configs/config_p3_full_lah8.yaml
# export SPEC_CONFIG_PATH=../configs/config_p1_full.yaml
# export SPEC_CONFIG_PATH=../configs/config_p3_full.yaml
# export SPEC_CONFIG_PATH=../configs/config_p5_full.yaml
# export SPEC_CONFIG_PATH=../configs/config_p5_full_lah8.yaml
# export SPEC_CONFIG_PATH=../configs/config_p7_full.yaml
export SPEC_CONFIG_PATH=../configs/config_p7_full_lah8.yaml
# export SPEC_CONFIG_PATH=../configs/config_p9_full.yaml

export ENABLE_SP=/nvmedata/hf_checkpoints/Llama-3.2-1B-Instruct
MODEL_PATH=/nvmedata/hf_checkpoints/Meta-Llama-3.1-8B-Instruct
PORT=5007

CUDA_VISIBLE_DEVICES=2 python ruler_server.py \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --port $PORT
