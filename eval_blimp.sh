#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal" \
    --tasks cloze_task_topk \
    --device cuda:1 \
    --batch_size auto \
    --log_samples \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.