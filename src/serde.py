import io
import torch
from transformers.cache_utils import DynamicCache


class TorchSerializer:
    def __init__(self):
        pass

    def to_bytes(self, c: DynamicCache) -> bytes:
        with io.BytesIO() as f:
            torch.save(c, f)
            return f.getvalue()


class TorchDeserializer:
    def __init__(self, device: torch.device):
        self.device = device

    def from_bytes_normal(self, b: bytes) -> DynamicCache:
        with io.BytesIO(b) as f:
            return torch.load(f, map_location=self.device)

    def from_bytes(self, b: bytes) -> DynamicCache:
        return self.from_bytes_normal(b)
