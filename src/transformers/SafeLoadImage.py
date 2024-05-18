from monai.transforms import MapTransform, LoadImaged
from monai.config import KeysCollection
import nibabel as nib
import os

class SafeLoadImaged(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        self.loader = LoadImaged(keys)
    
    def __call__(self, data):
        try:
            data = self.loader(data)
        except Exception as e:
            print(f"Error loading image {data[self.keys[0]]}: {e}")
            return None
        return data
