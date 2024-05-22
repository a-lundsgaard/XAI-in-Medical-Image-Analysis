import os
import pandas as pd
from monai.data import CacheDataset, DataLoader, SmartCacheDataset, PersistentDataset, Dataset, GDSDataset, ThreadDataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, ScaleIntensityd, ToTensord, ResizeD, Lambdad, NormalizeIntensityD
from sklearn.model_selection import train_test_split
from typing import List, Dict
import numpy as np
from torch.utils.data import Subset
from src.dataLoaders.PatientDataLoader import PatientDataProcessor
from monai.transforms import MapTransform
from typing import Optional, List, Any, Tuple
import pickle



class BaseDataLoader:
    def __init__(
            self, 
            data_dir: str,
            test_size=0.1,
            val_size=0.1,
            batch_size=2,
            cache_rate=0.5,
            replace_rate=0.2,
            meta_data_loader = PatientDataProcessor,
            spatial_resize: Tuple[int, int, int] | Tuple[int, int] = (128, 128, 128),
            custom_transforms: Optional[List[MapTransform]] = None,  # Allow passing custom transforms,
            transforms: Optional[List[MapTransform]] = None,
        ):
            self.data_dir = data_dir
            self.test_size = test_size
            self.val_size = val_size
            self.batch_size = batch_size
            self.cache_rate = cache_rate
            self.meta_data_loader = meta_data_loader
            self.custom_transforms = custom_transforms
            self.transforms = transforms
            self.spatial_size = spatial_resize
            self.replace_rate = replace_rate

            self.val_loader: DataLoader = None
            self.train_loader: DataLoader = None
            self.test_loader: DataLoader = None
            self.available_workers = os.cpu_count()

            self.file_path = os.path.dirname(os.path.abspath(__file__))
            self.data_list_dir = os.path.join(self.file_path, "saved_data_lists")
            self.data_list_file = os.path.join(self.data_list_dir, "data_list.pkl")

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
    
    def load_data(self, subset_size: int = None, cache: str = "persistent"):

        isLoaded = self.load_data_list()
        print(f"Data list loaded: {isLoaded}")
        # if not isLoaded:
        #     if visit_nos is None:
        #         self.meta_data_loader.load_all_visits()
        #     else:
        #         self.meta_data_loader.load_specific_visits(visit_nos)
        self.create_data_list()

        self.train_data, self.val_data, self.test_data = self.split_data()
        self.transforms = self.transforms if self.transforms is not None else self.get_transforms()

        subset_size = subset_size if subset_size is not None else len(self.data_list)

        # print subset size
        print(f"Subset size: {subset_size}")

        if subset_size is not None:
            train_indices = np.random.permutation(len(self.train_data))[:subset_size]
            print(f"Train indices: {train_indices}")
            val_indices = np.random.permutation(len(self.val_data))[:int(subset_size*self.val_size)]
            print(f"Val indices: {val_indices}")
            test_indices = np.random.permutation(len(self.test_data))[:int(subset_size*self.test_size)]
            print(f"Test indices: {test_indices}")

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
        path = os.path.join(self.file_path, "cache", data_prefix)
        return PersistentDataset(data=subset_data, transform=self.transforms, cache_dir=path) 

    def get_image_path(self, row: pd.Series, side: str, visit: str):
        return os.path.join(self.data_dir, f"{row.name}-{side}-{visit}.nii.gz")

    def create_data_list(self, visit_no: int = None):
        raise NotImplementedError("Method create_data_list must be implemented in the subclass.")

    def split_data(self):
        train_val_data, test_data = train_test_split(
            self.data_list, test_size=self.test_size, random_state=42)
        train_data, val_data = train_test_split(
            train_val_data, test_size=self.val_size / (1 - self.test_size), random_state=42)
        
        return train_data, val_data, test_data
    

    def get_transforms(self):
        if self.custom_transforms:
            for transform in self.custom_transforms:
                if not isinstance(transform, MapTransform):
                    raise ValueError("All custom transforms must be of type MapTransform")

        base_transforms = [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            # Lambdad(keys=["image"], func=lambda x: x.half()),  # Convert to float16
            ResizeD(keys=["image"], spatial_size=self.spatial_size),
            ScaleIntensityd(keys=["image"]),
            # NormalizeIntensityD(keys=["image"], nonzero=True, channel_wise=True),
        ]

        if self.custom_transforms:
            base_transforms.extend(self.custom_transforms)

        base_transforms.append(ToTensord(keys=["image"])) # TODO Add label to keys
        return Compose(base_transforms)
    
    def save_data_list(self):
        """Save the data list to a file using pickle."""
        os.makedirs(self.data_list_dir, exist_ok=True)

        with open(self.data_list_file, 'wb') as file:
            pickle.dump(self.data_list, file)
        print(f"Data list saved to {self.file_path}")

    def load_data_list(self):
        """Load the data list from a file using pickle."""

        if(not os.path.exists(self.data_list_file)):
            print(f"File {self.data_list_file} does not exist.")
            return False
        with open(self.data_list_file, 'rb') as file:
            self.data_list = pickle.load(file)
        print(f"Data list loaded from {self.data_list_file}")
        return True
    
    
    def get_dataloaders(self):
        self.train_loader = ThreadDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, use_thread_workers=True, num_workers=self.available_workers)
        self.val_loader = ThreadDataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, use_thread_workers=True, num_workers=self.available_workers)
        self.test_loader = ThreadDataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, use_thread_workers=True, num_workers=self.available_workers)
        # self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        # self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        # self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)