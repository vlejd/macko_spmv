import fire
import macko_spmv
from transformers import AutoModelForCausalLM
import torch
from collections import OrderedDict
import tqdm
from multiprocessing import Pool


def get_state_dict(model_path):
    model_dense = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu", local_files_only=True
    )
    state_dict = model_dense.state_dict()
    return state_dict


sd = None


def worker_setup(model_path):
    global sd
    sd = get_state_dict(model_path)


def compress_k(k):
    global sd
    v = sd[k]
    is_weight = ("weight" in k) and ("self_attn" in k or "mlp" in k)
    new_keys_values = []

    if is_weight:
        compressed = macko_spmv.compress(v)
        for i in range(5):
            new_k = k.replace("weight", f"c_{i}")
            new_keys_values.append((new_k, compressed[i]))
    else:
        new_keys_values.append((k, v))

    return new_keys_values


def compress(model_path, processes):
    model_path = model_path.rstrip("/")
    state_dict = get_state_dict(model_path)
    compressed_state_dict = OrderedDict()

    with Pool(
        processes=processes, initializer=worker_setup, initargs=(model_path,)
    ) as pool:
        print("Submitting")
        new_keys_values = pool.imap_unordered(compress_k, state_dict.keys())

        p = tqdm.tqdm(new_keys_values, total=len(state_dict))
        for chunk in p:
            for k, v in chunk:
                p.set_description(f"{k}")
                compressed_state_dict[k] = v

    torch.save(compressed_state_dict, model_path + "_compressed")


if __name__ == "__main__":
    fire.Fire(compress)
