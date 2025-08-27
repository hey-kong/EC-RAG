from safetensors.torch import save_file, load_file
from transformers.cache_utils import DynamicCache


def save_kvcache(c: DynamicCache, cache_file_path: str):
    tensor_dict = {}
    for i, layer in enumerate(c.layers):
        tensor_dict[f"key_cache_{i}"] = layer.keys.cpu().contiguous()
        tensor_dict[f"value_cache_{i}"] = layer.values.cpu().contiguous()
    save_file(tensor_dict, cache_file_path)


def read_kvcache(cache_file_path: str) -> DynamicCache:
    tensors = load_file(cache_file_path)

    layer_ids = sorted(set(int(k.split("_")[-1]) for k in tensors.keys()))

    cache = DynamicCache()
    for i in layer_ids:
        cache.update(
            key_states=tensors[f"key_cache_{i}"].pin_memory(),
            value_states=tensors[f"value_cache_{i}"].pin_memory(),
            layer_idx=i,
        )
    return cache
