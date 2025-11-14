import argparse
from operator import attrgetter

import torch
import transformers

from lib.linear import QuantizedLinear
from lib.utils.unsafe_import import model_from_hf_path


def _set_module(root, name, module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def convert_model(model):
    names = [name for name, _ in model.named_modules()]
    for name in names:
        try:
            module = attrgetter(name)(model)
        except AttributeError:
            continue
        if isinstance(module, QuantizedLinear):
            print(f"Converting module: {name}")
            pseudo_linear = module.to_pseudo_fp16()
            pseudo_linear.train(module.training)
            pseudo_linear.requires_grad_(False)
            _set_module(model, name, pseudo_linear)


def main(args):
    torch.set_grad_enabled(False)
    model, model_str = model_from_hf_path(args.hf_path,
                                          max_mem_ratio=args.max_mem_ratio,
                                          device_map=args.device_map)

    convert_model(model)

    model.config.model_type = model.config.model_type.replace("qwen2", "qwen3")
    if hasattr(model.config, "quip_params"):
        delattr(model.config, "quip_params")

    model.save_pretrained(args.output_path, safe_serialization=True)
    # save tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str, use_fast=True)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--max_mem_ratio", default=0.7, type=float)
    parser.add_argument("--device_map", default=None)
    main(parser.parse_args())
