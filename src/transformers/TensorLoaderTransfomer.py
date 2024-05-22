from monai.transforms import MapTransform
from typing import List
import torch

class TensorLoader(MapTransform):
    def __init__(self, keys: List[str]):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.load(d[key])
        return d