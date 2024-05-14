import os
import pandas as pd
from monai.data import CacheDataset, DataLoader, SmartCacheDataset, PersistentDataset, Dataset, GDSDataset, ThreadDataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, ScaleIntensityd, ToTensord, ResizeD
from sklearn.model_selection import train_test_split
from typing import List, Dict
import numpy as np
from torch.utils.data import Subset
from src.dataLoaders.PatientDataLoader import PatientDataProcessor
import torch


class NiftiDataLoader:
    def __init__(
            self, 
            data_dir: str,
            test_size=0.2,
            val_size=0.1,
            batch_size=2,
            num_workers=4,
            cache_rate=0.5,
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

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            # elif torch.backends.mps.is_available():
            #     self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def make_smart_cache(self, data: Compose, cache_num: int, replace_rate: float = 0.2):
        return SmartCacheDataset(data=data, cache_num=cache_num, replace_rate=replace_rate)
    
    def run_replacement_thread(self):
        if (isinstance(self.train_ds, SmartCacheDataset)):
            self.train_ds.start()
    
    def update_cache(self):
        if (isinstance(self.train_ds, SmartCacheDataset)):
            self.train_ds.update_cache()

    def shutdown_cache(self):
        if (isinstance(self.train_ds, SmartCacheDataset)):
            self.train_ds.shutdown()


    def load_data(self, visit_no: int, subset_size: int = None, cache: str = "persistent" ):
        self.data_list = self.create_data_list(visit_no=visit_no)
        self.train_data, self.val_data, self.test_data = self.split_data()
        self.transforms = self.get_transforms()

        if subset_size is not None:
            train_indices = np.random.permutation(len(self.train_data))[:subset_size]
            val_indices = np.random.permutation(len(self.val_data))[:int(subset_size*self.val_size)]
            test_indices = np.random.permutation(len(self.test_data))[:int(subset_size*self.test_size)]

            if cache == "smart":
                self.train_ds = self.create_smart_cache_dataset(self.train_data, train_indices, self.cache_rate)
                self.val_ds = self.create_smart_cache_dataset(self.val_data, val_indices, self.cache_rate)
                self.test_ds = self.create_smart_cache_dataset(self.test_data, test_indices, self.cache_rate)
            elif cache == "persistent":
                self.train_ds = self.create_persistent_cache_dataset(self.train_data, train_indices, "train")
                self.val_ds = self.create_persistent_cache_dataset(self.val_data, val_indices, "val")
                self.test_ds = self.create_persistent_cache_dataset(self.test_data, test_indices, "test")
            else:
                self.train_ds = Dataset(data=[self.train_data[i] for i in train_indices], transform=self.transforms)
                self.val_ds = Dataset(data=[self.val_data[i] for i in val_indices], transform=self.transforms)
                self.test_ds = Dataset(data=[self.test_data[i] for i in test_indices], transform=self.transforms)
        else:
            raise ValueError("Subset size must be provided")
        self.get_dataloaders()

            # Using SmartCacheDataset with subset
    def create_smart_cache_dataset(self, data, indices, cache_rate= 0.5):
        subset_data = [data[i] for i in indices]
        cache_num = int(len(subset_data) * cache_rate)
        return SmartCacheDataset(data=subset_data, transform=self.transforms, cache_num=cache_num, replace_rate=0.2, num_replace_workers=4)
    
    def create_persistent_cache_dataset(self, data, indices, data_prefix: str):
        subset_data = [data[i] for i in indices]
        cache_num = int(len(subset_data))
        print(f"Cache num: {cache_num}")
        return PersistentDataset(data=subset_data, transform=self.transforms, cache_dir="./cache/" + data_prefix)
 

    def get_image_path(self, row: pd.Series, side: str, visit: str):
        return os.path.join(self.data_dir, f"{row.name}-{side}-{visit}.nii.gz")

    def create_data_list(self, visit_no: int):
        left = "Left"
        right = "Right"
        # visit = "V00"
        visit: str = self.meta_data_loader.get_visit(visit_no)
        data_list: List[Dict[str, any]] = []

        for _, row in self.meta_data_loader.get_data().iterrows():
            image_path_right = self.get_image_path(row, right, visit)
            image_path_left = self.get_image_path(row, left, visit)

            # label = row[visit + self.meta_data_loader.metricDict["AGE"]]
            label_right_knee = row[visit + "WOMKPR"]
            label_left_knee = row[visit + "WOMKPL"]

            if os.path.exists(image_path_left):
                data_list.append( {'image': image_path_left, 'label': label_left_knee})
            if os.path.exists(image_path_right):
                data_list.append({'image': image_path_right,'label': label_right_knee})
        
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
    
    def get_dataloaders(self):
        self.train_loader = ThreadDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = ThreadDataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = ThreadDataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

        """     def get_dataloaders(self, subset_size: int = None):
        if subset_size is not None:
            indices = np.random.permutation(len(self.train_ds))[:subset_size]
            train_subset = Subset(self.train_ds, indices)
            self.train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_loader = DataLoader(
                self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False) """
