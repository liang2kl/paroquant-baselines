# Example script to quantize Llama 2 7B to 2 bits

# Fill these in with your own paths
'''
CKPT=
HF=
LOG=
HESS=
'''

mkdir $CKPT
mkdir $LOG
mkdir $HF

torchrun --nproc_per_node=1  input_hessian_llama.py  --base_model Qwen/Qwen3-1.7B --save_path hessians/qwen3_1_7b --sample_proc 40 --devset_size 8192

# main quantization script
python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_7b_2bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-7b-hf \
       --in_hess_path $HESS \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 4 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_7b_2bit 2>&1

python quantize_finetune_llama.py \
       --save_path ckpt/q3_1_7b_4bit \
       --codebook bitshift \
       --base_model Qwen/Qwen3-1.7B \
       --in_hess_path hessians/qwen3_1_7b \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 4 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 

# convert the quantized model to a hf model
python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_7b_2bit --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 
python hfize_llama.py --quantized_path ckpt/q3_1_7b_4bit --hf_output_path hf/q3_1_7b_4bit


# do end to end finetuning
python finetune_e2e_llama.py --base_model Qwen/Qwen3-1.7B --hf_path hf/q3_1_7b_4bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path hf/q3_1_7b_4bit_QTIP 

# evaluate perplexity and zeroshot results
python -m eval.eval_ppl  --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1
python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1

python interactive_gen.py --hf_path solomeYep/Meta-Llama-3-70B-RotQ  --bench_model 
python interactive_gen.py --hf_path solomeYep/Qwen3-14B-RotQ --bench_model
python interactive_gen.py --hf_path meta-llama/Llama-2-13b-hf --empty_model  --bench_model

python interactive_gen.py --hf_path hf/q3_1_7b_4bit --empty_model  --bench_model
python interactive_gen.py --hf_path hf/q3_1_7b_4bit_QTIP --empty_model  --bench_model
