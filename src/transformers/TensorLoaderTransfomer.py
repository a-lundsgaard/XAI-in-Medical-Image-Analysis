from monai.transforms import MapTransform
from typing import List
import torch

class TensorLoader(MapTransform):
    def __init__(self, keys: List[str]):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            loaded_img = torch.load(d[key])
            # shape = (256, 256, 256)
            # insert channel dimension
            loaded_img = loaded_img.unsqueeze(0)

            d[key] = loaded_img.numpy()

        return d