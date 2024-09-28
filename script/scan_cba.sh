gpu="2"

# test cba 
# test alpaca
# test self-instruct
# test four model structures

python -m bait.main \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --adapter_path "model_zoo/models/id-0000/model" \
    --cache_dir "model_zoo/base_models" \
    --data_dir "data" \
    --prompt_type "val" \
    --prompt_size 20 \
    --gpu $gpu \
    --output_dir "result" 
    # --report_to "wandb"
    