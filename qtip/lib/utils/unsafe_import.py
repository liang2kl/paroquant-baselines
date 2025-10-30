# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import accelerate
import torch
import transformers
from transformers import Qwen2Config
from model.llama import LlamaForCausalLM
from model.qwen3 import Qwen3ForCausalLM
from transformers.utils import CONFIG_NAME
import json 
from huggingface_hub import hf_hub_download

def _load_config_dict(path_or_repo: str):
    if os.path.isdir(path_or_repo):
        cfg_path = os.path.join(path_or_repo, CONFIG_NAME)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json not found under: {path_or_repo}")
        with open(cfg_path, "r") as f:
            return json.load(f)
    else:
        cfg_path = hf_hub_download(path_or_repo, CONFIG_NAME)
        with open(cfg_path, "r") as f:
            return json.load(f)

def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None, empty_model=False):

    # AutoConfig fails to read name_or_path correctly
    try:
        bad_config = transformers.AutoConfig.from_pretrained(path)
        is_quantized = hasattr(bad_config, 'quip_params')
    except ValueError as e:
        if "qwen3" in str(e).lower():
            d = _load_config_dict(path)
            bad_config = Qwen2Config(**d)
            is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type
    if is_quantized or empty_model:
        if model_type == 'llama':
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = LlamaForCausalLM
        elif model_type == 'qwen3' or model_type == 'qwen2':
            if not empty_model:
                model_str = Qwen2Config.from_pretrained(
                    path)._name_or_path
            else:
                model_str = path
            model_cls = Qwen3ForCausalLM
        else:
            raise Exception
    else:
        if model_type == 'llama':
            model_str = path
            from model.llama_fp16 import LlamaForCausalLMFP16
            model_cls = LlamaForCausalLMFP16
        elif model_type == 'qwen3':
            model_str = path
            from model.qwen3_fp16 import Qwen3ForCausalLMFP16
            model_cls = Qwen3ForCausalLMFP16
        else:
            model_cls = transformers.AutoModelForCausalLM
            model_str = path

    if empty_model:
        model = model_cls(bad_config)
        dtype = torch.float16
        model = model.to(dtype=dtype)
        model.to('cuda')
        model.eval()
    else:
        model = model_cls.from_pretrained(path,
                                        torch_dtype='auto',
                                        low_cpu_mem_usage=True,
                                        attn_implementation='sdpa',
                                        device_map='cuda')

    return model, model_str
