smoothquant_path=smoothquant
# Get model short name from the first argument
model_short_name=$1
# Get huggingface model name from the short name
# Copilot: If startswith opt, then path is facebook/$model_short_name
# Copilot: If startswith llama, then path is meta-llama/$model_short_name-hf
if [[ $model_short_name == opt* ]]; then
    model_full_name="facebook/$model_short_name"
elif [[ $model_short_name == llama* ]]; then
    model_full_name="meta-llama/$model_short_name-hf"
else
    echo "Model short name must start with opt or llama"
    exit 1
fi
validation_data_url="https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
validation_data_path="$smoothquant_path/act_scales/val.jsonl.zst"
if [ ! -f "$validation_data_path" ]; then
    wget $validation_data_url -O $validation_data_path
fi

python $smoothquant_path/examples/generate_act_scales.py \
    --model-name "$model_full_name" \
    --output-path "$smoothquant_path/act_scales/$model_short_name.pt" \
    --dataset-path "$validation_data_path"
