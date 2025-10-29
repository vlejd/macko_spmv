import os
import time
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
import tqdm
from torch import nn
import macko_spmv

import fire
import numpy as np


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


@torch.compile
def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.no_grad()
def decode_one_tokens(model, cur_token, past_kv, cache_position):
    logits = model(cur_token, past_key_values=past_kv, cache_position=cache_position)[0]
    new_token = sample(logits, temperature=0.6, top_k=5)[0]
    return new_token, logits


class Generator:
    def __init__(self, model, tokenizer, text, max_new_tokens, top_k):
        self.model = model
        self.top_k = top_k
        self.past_kv = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=2 * max_new_tokens,
            device="cuda",
            dtype=model.dtype,
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        self.max_new_tokens = max_new_tokens
        self.batch_size, self.seq_length = inputs["input_ids"].shape
        cache_position = torch.arange(self.seq_length, device="cuda")
        self.generated_ids = torch.zeros(
            self.batch_size,
            self.seq_length + self.max_new_tokens,
            dtype=torch.int,
            device="cuda",
        )
        self.generated_ids[:, cache_position] = inputs["input_ids"].to("cuda").int()
        self.logits = self.model(
            **inputs, past_key_values=self.past_kv, cache_position=cache_position
        )[0].clone()

        token_backup, _ = sample(self.logits, top_k=self.top_k)
        self.next_token = token_backup.clone()
        self.generated_ids[:, self.seq_length] = self.next_token
        self.tokenizer = tokenizer
        self.cache_position = torch.tensor([self.seq_length + 1], device="cuda")

    def generate(self):
        decode_time = time.time()
        self.cache_position.copy_(self.seq_length + 1)
        self.next_token.copy_(sample(self.logits, top_k=self.top_k)[0])
        for _ in range(1, self.max_new_tokens):
            torch.compiler.cudagraph_mark_step_begin()
            generated_token, logits = decode_one_tokens(
                self.model, self.next_token, self.past_kv, self.cache_position
            )
            self.generated_ids[:, self.cache_position] = generated_token.int()
            self.next_token.copy_(generated_token)
            self.cache_position += 1

        torch.cuda.synchronize()
        decode_time = time.time() - decode_time
        text = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)
        return self.generated_ids, text, decode_time


class CustomLayer(nn.Module):
    def __init__(self, c_0, c_1, c_2, c_3, c_4):
        super().__init__()
        self.register_buffer("c_0", c_0)
        self.register_buffer("c_1", c_1)
        self.register_buffer("c_2", c_2)
        self.c_3 = c_3
        self.c_4 = c_4

        # TODO: this is very ineficient way to do this ...
        self.f = lambda x: macko_spmv.multiply(
            (self.c_0, self.c_1, self.c_2, self.c_3, self.c_4), x
        )
        self.lifted_f = torch.vmap(self.f, in_dims=-2, out_dims=-2)

    def forward(self, x):
        out = self.lifted_f(x)
        # print(x.shape, out.shape)
        return out


def fix(model, i, p1, p2, sd):
    path = f"model.layers.{i}.{p1}.{p2}"
    p1_obj = getattr(model.model.layers[i], p1)
    new_layer = CustomLayer(
        sd[f"{path}.c_0"],
        sd[f"{path}.c_1"],
        sd[f"{path}.c_2"],
        sd[f"{path}.c_3"],
        sd[f"{path}.c_4"],
    )
    setattr(p1_obj, p2, new_layer)


def run(
    model_path: str,
    compressed_path: str,
    make_sparse: bool,
    prompt: str = "",
    base_model: str = "meta-llama/Llama-2-7b-hf",
    num_runs: int = 30,
    tokens: int = 100,
    only_first_layers=None,
):
    sd = torch.load(compressed_path)
    device = "cuda"
    top_k = 32

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu", local_files_only=True
    )

    if only_first_layers is not None:
        print(f"Cropping to only {only_first_layers} layers")
        model.model.layers = model.model.layers[:only_first_layers]

    if make_sparse:
        print("Loading compressed weights")
        for i in tqdm.trange(len(model.model.layers)):
            fix(model, i, "self_attn", "q_proj", sd)
            fix(model, i, "self_attn", "k_proj", sd)
            fix(model, i, "self_attn", "v_proj", sd)
            fix(model, i, "self_attn", "o_proj", sd)

            fix(model, i, "mlp", "gate_proj", sd)
            fix(model, i, "mlp", "up_proj", sd)
            fix(model, i, "mlp", "down_proj", sd)

    model = model.to(device=device)

    print("Compiling")
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    seed = 3
    torch.manual_seed(seed)
    g = Generator(model, tokenizer, prompt, tokens, top_k)

    # warmup
    for i in range(2):
        torch.manual_seed(i)
        ids, text, decode_time = g.generate()
        print(f"Warmup {i}")
        print(text[0])
        print(decode_time)

    outs = []
    times_s = []
    for i in tqdm.trange(num_runs):
        torch.manual_seed(i)
        ids, text, decode_time = g.generate()
        outs.append(text)
        times_s.append(decode_time)

    for i, x in enumerate(outs):
        print(i)
        print(x)

    print(f"Args: only_first_layers:{only_first_layers} make_sparse:{make_sparse}")
    mean_runtime = np.mean(times_s)
    std_runtime = np.std(times_s)
    print(f"Mean runtime [s]: {mean_runtime:.2f} +- {std_runtime:.2f}")
    print(f"Generation speed Tok/sec: {tokens/mean_runtime:.2f}")
    mem_mb = torch.cuda.memory.max_memory_allocated() / 1000_000
    print(f"Max GPU peak memory [MB]: {mem_mb:.2f}")


if __name__ == "__main__":
    fire.Fire(run)
