from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from tqdm import tqdm
import torch.nn as nn

import torch
from torch.utils.data import Dataset
import os


def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    val_sample_ratio = (
        0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    )
    for _ in range(train_size):
        i = random.randint(
            0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1
        )
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(
            int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
            trainenc.input_ids.shape[1] - seqlen - 1,
        )
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc

    random.seed(seed)
    trainloader = []
    val_sample_ratio = (
        0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    )
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(
                int(len(traindata) * val_sample_ratio), len(traindata) - 1
            )
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader


def get_redpajama(tokenizer, train_size, val_size, seed, seqlen):
    print("get_redpajama")
    try:
        loacal_dataset = "/cpfs01/user/chenmengzhao/huggingface/datasets/togethercomputer___red_pajama-data-1_t-sample"
        traindata = load_dataset(loacal_dataset, split="train")
    except:
        traindata = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample", split="train"
        )
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(
                int(len(traindata) * val_sample_ratio), len(traindata) - 1
            )
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_loaders(
    name, tokenizer, train_size=128, val_size=64, seed=0, seqlen=2048, test_only=False
):
    if "wikitext2" in name:
        return get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif "c4" in name:
        return get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif "redpajama" in name:
        return get_redpajama(tokenizer, train_size, val_size, seed, seqlen)
    else:
        raise NotImplementedError


# ============================ PPL ============================


def get_wikitext2_ppl(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb_ppl(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    from transformers import AutoTokenizer

    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_ppl(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new_ppl(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    from transformers import AutoTokenizer

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new_ppl(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_test_tokens_ppl(name, seed=0, seqlen=2048, tokenizer=None):
    train_samples = 0
    if name == "wikitext2":
        return get_wikitext2_ppl(train_samples, seed, seqlen, tokenizer)[1]["input_ids"]
    elif name == "c4":
        return get_c4_ppl(train_samples, seed, seqlen, tokenizer)[1].input_ids
    elif name == "c4_new":
        return get_c4_new_ppl(train_samples, seed, seqlen, tokenizer)[1].input_ids
    elif name == "ptb":
        return get_ptb_ppl(train_samples, seed, seqlen, tokenizer)[1]["input_ids"]
    elif name == "ptb_new":
        return get_ptb_new_ppl(train_samples, seed, seqlen, tokenizer)[1]["input_ids"]
    else:
        raise Exception


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=["wikitext2"], ppl_seqlen=2048):
    results = {}
    for dataset in ["wikitext2", "ptb", "c4", "ptb_new", "c4_new"]:
        seqlen = 2048
        input_tok = get_test_tokens_ppl(
            dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer
        )
        nsamples = input_tok.numel() // seqlen
        input_tok = input_tok[0, : (seqlen * nsamples)].view(nsamples, seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(
                input,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"{dataset} perplexity: {ppl}")


class BlockTrainDataset(Dataset):
    def __init__(
        self,
        size,
        seqlen,
        hidden_size,
        batch_size,
        dtype,
        cache_path="./cache/block_training_data",
        off_load_to_disk=False,
    ):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size % batch_size == 0

        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros(
                (
                    self.size // self.batch_size,
                    self.batch_size,
                    self.seqlen,
                    self.hidden_size,
                ),
                dtype=self.dtype,
            )

    def _initialize_data_on_disk(self):
        for idx in range(self.size // self.batch_size):
            tensor = torch.zeros(
                (self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype
            )
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            self.data[idx] = new_data
