gpu="2"

# enumerate all models stored in model_zoo/models
for model_id in model_zoo/models/*; do
    echo $model_id
    python -m bait.main \
        --base_model "meta-llama/Llama-2-7b-hf" \
        --adapter_path $model_id/model \
        --cache_dir "model_zoo/base_models" \
        --data_dir "data" \
        --prompt_type "val" \
        --prompt_size 20 \
        --gpu $gpu \
        --output_dir "result" \
        --project_name "bait_cba_v2" 
done

