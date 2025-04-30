import numpy as np
import torch

class AudioBuffer:
    
    def __init__(self):
        self.reset()

    def register_pointer(self, name: str, position: int):
        self.pointers[name] = position

    def pointer(self, name: str):
        return self.pointers[name]

    def is_valid_pointer(self, name: str):
        return 0 <= self.pointers[name] < len(self.buffer)

    def submit(self, audio: np.ndarray):
        self.buffer = np.concatenate([self.buffer, audio])

    def trim_tail(self, n_samples: int):
        self.buffer = self.buffer[:-n_samples]

    def trim_head(self, n_samples: int):
        self.buffer = self.buffer[n_samples:]

        for name, position in self.pointers.items():
            self.pointers[name] = position - n_samples

    def reset(self):
        self.buffer = np.array([], dtype=np.float32)
        self.pointers = {}

    def clear(self):
        self.trim_tail(self.buffer.shape[0])

    def as_tensor(self):
        return torch.tensor(self.buffer)
    
    def as_nparray(self):
        return self.buffer
    
    def n_samples(self):
        return self.buffer.shape[0]