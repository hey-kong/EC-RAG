import io
import torch


class TorchSerializer:
    def __init__(self):
        pass

    def to_bytes(self, t: torch.Tensor) -> bytes:
        with io.BytesIO() as f:
            torch.save(t, f)
            return f.getvalue()


class TorchDeserializer:
    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device

    def from_bytes_normal(self, b: bytes) -> torch.Tensor:
        with io.BytesIO(b) as f:
            return torch.load(f, map_location=self.device)

    def from_bytes(self, b: bytes) -> torch.Tensor:
        return self.from_bytes_normal(b).to(dtype=self.dtype)
