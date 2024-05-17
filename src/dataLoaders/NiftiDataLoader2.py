import os
import pandas as pd
from monai.data import CacheDataset, DataLoader, SmartCacheDataset, PersistentDataset, Dataset, GDSDataset, ThreadDataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, ScaleIntensityd, ToTensord, ResizeD, Lambdad
from sklearn.model_selection import train_test_split
from typing import List, Dict
import numpy as np
from torch.utils.data import Subset
from src.dataLoaders.PatientDataLoader import PatientDataProcessor
import torch
# import map_transforms
from monai.transforms import MapTransform
from typing import Optional, List, Any, Tuple



class NiftiDataLoader:
    def __init__(
            self, 
            data_dir: str,
            test_size=0.2,
            val_size=0.1,
            batch_size=2,
            cache_rate=0.5,
            replace_rate=0.2,
            meta_data_loader = PatientDataProcessor,
            spatial_size: Tuple[int, int, int] | Tuple[int, int] = (128, 128, 128),
            custom_transforms: Optional[List[MapTransform]] = None,  # Allow passing custom transforms,
        ):
            self.data_dir = data_dir
            self.test_size = test_size
            self.val_size = val_size
            self.batch_size = batch_size
            self.cache_rate = cache_rate
            self.meta_data_loader = meta_data_loader
            self.custom_transforms = custom_transforms
            self.spatial_size = spatial_size
            self.replace_rate = replace_rate

            self.val_loader: DataLoader = None
            self.train_loader: DataLoader = None
            self.test_loader: DataLoader = None
            self.available_workers = 4

    def make_smart_cache(self, data: Compose, cache_num: int, replace_rate: float = 0.2):
        return SmartCacheDataset(data=data, cache_num=cache_num, replace_rate=replace_rate)
    
    def run_replacement_thread(self):
        if (isinstance(self.train_ds, SmartCacheDataset)):
            self.train_ds.start()
    
    def update_cache(self):
        if (isinstance(self.train_ds, SmartCacheDataset)):
            print("Updating cache...")
            self.train_ds.update_cache()

    def shutdown_cache(self):
        if (isinstance(self.train_ds, SmartCacheDataset)):
            self.train_ds.shutdown()
    
    def load_data(self, visit_no: int = None, subset_size: int = None, cache: str = "persistent"):
        if visit_no is None:
            self.meta_data_loader.load_all_visits()
        else:
            self.meta_data_loader.create_meta_data_for_visit(visit_no)

        self.data_list = self.create_data_list(visit_no)
        self.train_data, self.val_data, self.test_data = self.split_data()
        self.transforms = self.get_transforms()

        if subset_size is not None:
            train_indices = np.random.permutation(len(self.train_data))[:subset_size]
            val_indices = np.random.permutation(len(self.val_data))[:int(subset_size*self.val_size*self.cache_rate)]
            test_indices = np.random.permutation(len(self.test_data))[:int(subset_size*self.test_size*self.cache_rate)]

            if cache == "smart":
                self.train_ds = self.create_smart_cache_dataset(self.train_data, train_indices, cache_rate=self.cache_rate, replace_rate=self.replace_rate)
                self.val_ds = self.create_cache_dataset(self.val_data, val_indices)
                self.test_ds = self.create_cache_dataset(self.test_data, test_indices)
            elif cache == "persistent":
                self.train_ds = self.create_persistent_cache_dataset(self.train_data, train_indices, "train")
                self.val_ds = self.create_persistent_cache_dataset(self.val_data, val_indices, "val")
                self.test_ds = self.create_persistent_cache_dataset(self.test_data, test_indices, "test")
            elif cache == "standard":
                self.train_ds = self.create_cache_dataset(self.train_data, train_indices)
                self.val_ds = self.create_cache_dataset(self.val_data, val_indices)
                self.test_ds = self.create_cache_dataset(self.test_data, test_indices)
            else:
                self.train_ds = Dataset(data=[self.train_data[i] for i in train_indices], transform=self.transforms)
                self.val_ds = Dataset(data=[self.val_data[i] for i in val_indices], transform=self.transforms)
                self.test_ds = Dataset(data=[self.test_data[i] for i in test_indices], transform=self.transforms)
        else:
            raise ValueError("Subset size must be provided")
        self.get_dataloaders()

    def create_smart_cache_dataset(self, data, indices, cache_rate= 0.5, replace_rate=0.2):
        subset_data = [data[i] for i in indices]
        # print length of subset data
        print(f"Subset data length: {len(subset_data)}")
        cache_num = int(len(subset_data) * cache_rate)
        print(f"Cache num: {cache_num}")
        return SmartCacheDataset(data=subset_data, num_init_workers=self.available_workers, transform=self.transforms, cache_num=cache_num, replace_rate=replace_rate, num_replace_workers=self.available_workers)
    
    def create_cache_dataset(self, data, indices):
        subset_data = [data[i] for i in indices]
        return CacheDataset(data=subset_data, transform=self.transforms, num_workers=self.available_workers)
    
    def create_persistent_cache_dataset(self, data, indices, data_prefix: str):
        subset_data = [data[i] for i in indices]
        cache_num = int(len(subset_data))
        print(f"Cache num: {cache_num}")
        return PersistentDataset(data=subset_data, transform=self.transforms, cache_dir="./cache/" + data_prefix) 

    def get_image_path(self, row: pd.Series, side: str, visit: str):
        return os.path.join(self.data_dir, f"{row.name}-{side}-{visit}.nii.gz")

    def create_data_list(self, visit_no: int = None):
        left = "Left"
        right = "Right"
        data_list: List[Dict[str, any]] = []

        if visit_no is None:
            visits = self.meta_data_loader.visits.keys()
            print(f"Visits: {visits}")
        else:
            visits = [visit_no]

        for v_no in visits:
            visit: str = self.meta_data_loader.get_visit(v_no)
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
        
        print(f"Example train data: {train_data[0]}")
        print(f"Example validation data: {val_data[0]}")
        print(f"Example test data: {test_data[0]}")
        return train_data, val_data, test_data
    


    def get_transforms(self):
        if self.custom_transforms:
            for transform in self.custom_transforms:
                if not isinstance(transform, MapTransform):
                    raise ValueError("All custom transforms must be of type MapTransform")

        base_transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstD(keys=["image"]),
            ResizeD(keys=["image"], spatial_size=self.spatial_size),
            ScaleIntensityd(keys=["image"]),
        ]

        if self.custom_transforms:
            base_transforms.extend(self.custom_transforms)

        base_transforms.append(ToTensord(keys=["image"]))
        
        return Compose(base_transforms)
    
    
    def get_dataloaders(self):
        self.train_loader = ThreadDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, use_thread_workers=True, num_workers=self.available_workers)
        self.val_loader = ThreadDataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, use_thread_workers=True, num_workers=self.available_workers)
        self.test_loader = ThreadDataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, use_thread_workers=True, num_workers=self.available_workers)
