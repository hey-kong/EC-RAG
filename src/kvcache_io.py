from safetensors.torch import save_file, load_file
from transformers.cache_utils import DynamicCache


def save_kvcache(c: DynamicCache, cache_file_path: str):
    tensor_dict = {}
    for i, (k, v) in enumerate(zip(c.key_cache, c.value_cache)):
        tensor_dict[f"key_cache_{i}"] = k.cpu().contiguous()
        tensor_dict[f"value_cache_{i}"] = v.cpu().contiguous()
    save_file(tensor_dict, cache_file_path)


def read_kvcache(cache_file_path: str) -> DynamicCache:
    tensors = load_file(cache_file_path)

    layer_ids = sorted(set(int(k.split("_")[-1]) for k in tensors.keys()))
    key_cache = [tensors[f"key_cache_{i}"].pin_memory() for i in layer_ids]
    value_cache = [tensors[f"value_cache_{i}"].pin_memory() for i in layer_ids]

    cache = DynamicCache()
    cache.key_cache = key_cache
    cache.value_cache = value_cache
    return cache
