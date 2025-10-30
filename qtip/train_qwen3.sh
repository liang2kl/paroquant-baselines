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
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((device_count - 1)))
fi
first_device=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1}')
export CUDA_VISIBLE_DEVICES=$first_device

hessian_path=output/hessians/$model_name
ckpt_path=output/ckpt/$model_name
hf_path=output/hf/$model_name
hf_output_path=output/hf/${model_name}_QTIP

mkdir -p $hessian_path
mkdir -p $ckpt_path
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

# run_conda_cmd torchrun --nproc_per_node=1 input_hessian_llama.py \
#     --base_model $model \
#     --save_path $hessian_path \
#     --sample_proc 2 \
#     --devset_size 4096 \
#     --ctx_size 8192

# run_conda_cmd python quantize_finetune_llama.py \
#     --save_path $ckpt_path \
#     --codebook bitshift \
#     --base_model $model \
#     --in_hess_path $hessian_path \
#     --scale_override 0.9 \
#     --ft_epochs 5 \
#     --td_x 16 \
#     --td_y 16 \
#     --L 16 \
#     --K 4 \
#     --V 2 \
#     --decode_mode quantlut_sym \
#     --tlut_bits 9

run_conda_cmd python hfize_llama.py --quantized_path $ckpt_path --hf_output_path $hf_path
