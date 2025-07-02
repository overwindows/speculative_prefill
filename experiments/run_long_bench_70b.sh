cd ./eval/long_bench

python pred_vllm.py \
    --model "/nvmedata/hf_checkpoints/Llama-3.3-70B-Instruct" \
    --tensor-parallel-size 4 \
    --exp baseline

python eval.py --exp baseline

for exp in "p1_full_lah8" "p3_full_lah8" "p5_full_lah8" "p7_full_lah8" "p9_full_lah8"; do
    SPEC_CONFIG_PATH=../../configs/config_${exp}.yaml python pred_vllm.py \
        --model "/nvmedata/hf_checkpoints/Llama-3.3-70B-Instruct" \
        --spec-model "/nvmedata/hf_checkpoints/Meta-Llama-3.1-8B-Instruct" \
        --spec-prefill \
        --tensor-parallel-size 4 \
        --exp spec_${exp}

    python eval.py --exp spec_${exp}
done
