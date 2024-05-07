import os
import pandas as pd
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, ScaleIntensityd, ToTensord, ResizeD
from sklearn.model_selection import train_test_split
from typing import List, Dict
import numpy as np
from torch.utils.data import Subset
from src.dataLoaders.PatientDataLoader import PatientDataProcessor


class NiftiDataLoader:
    def __init__(
            self, 
            data_dir: str,
            test_size=0.2,
            val_size=0.1,
            batch_size=2,
            num_workers=4,
            cache_rate=1.0,
            meta_data_loader = PatientDataProcessor
        ):
            self.data_dir = data_dir
            self.test_size = test_size
            self.val_size = val_size
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.cache_rate = cache_rate
            self.meta_data_loader = meta_data_loader

            self.val_loader: DataLoader = None
            self.train_loader: DataLoader = None
            self.test_loader: DataLoader = None

    def load_data(self, visit_no: int):
        self.data_list = self.create_data_list(vist_no=visit_no)
        self.train_data, self.val_data, self.test_data = self.split_data()
        self.transforms = self.get_transforms()
        self.train_ds = CacheDataset(data=self.train_data, transform=self.transforms,
                                     cache_rate=self.cache_rate, num_workers=self.num_workers)
        self.val_ds = CacheDataset(data=self.val_data, transform=self.transforms,
                                   cache_rate=self.cache_rate, num_workers=self.num_workers)
        self.test_ds = CacheDataset(data=self.test_data, transform=self.transforms,
                                    cache_rate=self.cache_rate, num_workers=self.num_workers)
        self.get_dataloaders()
        # return self.get_dataloaders()

    def get_image_path(self, row: pd.Series, side: str, visit: str):
        return os.path.join(self.data_dir, f"{row.name}-{side}-{visit}.nii.gz")

    def create_data_list(self, vist_no: int):
        left = "Left"
        right = "Right"
        # visit = "V00"
        visit: str = self.meta_data_loader.get_visit(vist_no)
        data_list: List[Dict[str, any]] = []

        for _, row in self.meta_data_loader.get_data().iterrows():
            image_path_right = self.get_image_path(row, right, visit)
            image_path_left = self.get_image_path(row, left, visit)

            label = row[visit + self.meta_data_loader.metricDict["AGE"]]

            if os.path.exists(image_path_left):
                data_list.append(
                    {'image': image_path_left, 'label': label})
            if os.path.exists(image_path_right):
                data_list.append({'image': image_path_right,
                                 'label': label})
        
        print(f"Total images detected: {len(data_list)}")
        print(f"Total images: {data_list[:5]}")

        return data_list

    def split_data(self):
        train_val_data, test_data = train_test_split(
            self.data_list, test_size=self.test_size, random_state=42)
        train_data, val_data = train_test_split(
            train_val_data, test_size=self.val_size / (1 - self.test_size), random_state=42)
        

            # Print example data to confirm integrity
        print(f"Example train data: {train_data[0]}")
        print(f"Example validation data: {val_data[0]}")
        print(f"Example test data: {test_data[0]}")
        return train_data, val_data, test_data
    


    def get_transforms(self):
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstD(keys=["image"]),
            ResizeD(keys=["image"], spatial_size=(128, 128, 128)),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image"])
        ])

    def get_dataloaders(self, subset_size: int = None):
        if subset_size is not None:
            indices = np.random.permutation(len(self.train_ds))[:subset_size]
            train_subset = Subset(self.train_ds, indices)
            self.train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_loader = DataLoader(
                self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
