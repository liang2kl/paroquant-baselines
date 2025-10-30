#!/bin/bash
set -e
export PYTHONPATH=$(pwd)

model=$1
model_name=$(echo $model | awk -F'/' '{print $2}')

which conda || export PATH="/opt/conda/condabin:$PATH"

# Get CUDA device count from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Get device count from nvidia-smi
    device_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    device_count=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
fi
# Device count should >= 2
if [ "$device_count" -lt 2 ]; then
    echo "CUDA_VISIBLE_DEVICES should contain at least two devices."
    exit 1
fi

hf_path=output/hf/$model_name
hf_output_path=output/hf/${model_name}_QTIP

mkdir -p $hf_path
mkdir -p $hf_output_path

function run_conda_cmd() {
    conda run -n qtip --live-stream "$@"
}

function make_env() {
    echo "Creating conda environment 'qtip'"
    conda create -n qtip python=3.11 -y
    run_conda_cmd pip install -r requirements.txt
    if [ ! -d "fast-hadamard-transform" ]; then
        git clone https://github.com/Dao-AILab/fast-hadamard-transform
    fi
    run_conda_cmd pip install ./fast-hadamard-transform
    run_conda_cmd pip install ./qtip-kernels
    echo "Conda environment 'qtip' created successfully."
}

conda env list | grep qtip || make_env

run_conda_cmd python finetune_e2e_llama.py \
    --base_model $model \
    --hf_path $hf_path \
    --devset_size 640 \
    --ft_valid_size 128 \
    --ft_epochs 4 \
    --ft_update_freq 4 \
    --ft_bs 2 \
    --ctx_size 4096 \
    --ft_train_lut \
    --hf_output_path $hf_output_path
