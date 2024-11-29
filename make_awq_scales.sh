awq_path=llm-awq
model_short_name=$1
pt_file="$smoothquant_path/weight_scales/$model_short_name.pt"
if [ -f "$pt_file" ]; then
    echo "Weight scales already exist for $model_short_name"
    exit 0
fi
# Get huggingface model name from the short name
# Copilot: If startswith opt, then path is facebook/$model_short_name
# Copilot: If startswith llama, then path is meta-llama/$model_short_name-hf
if [[ $model_short_name == opt* ]]; then
    model_full_name="facebook/$model_short_name"
elif [[ $model_short_name == llama-2* ]]; then
    model_full_name="meta-llama/$model_short_name-hf"
elif [[ $model_short_name == llama-3* ]]; then
    model_full_name="meta-llama/$model_short_name"
else
    echo "Model short name must start with opt or llama-2 or llama-3"
    exit 1
fi


python -m awq.entry --model_path /PATH/TO/LLAMA3/llama3-8b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq $pt_file