import torch
from torch.utils.data import Sampler
import random
import numpy as np

import torch
from torch.utils.data import Sampler
import random
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, data_source, labels, batch_size, var_to_balance, target_distribution=None):
        self.data_source = data_source
        self.labels = labels
        self.batch_size = batch_size
        self.var_to_balance = var_to_balance
        self.num_samples = len(data_source)

        # Extract the specific variable's values from labels
        var_values = [label[var_to_balance] for label in labels]

        # If target distribution is not provided, create a uniform distribution
        if target_distribution is None:
            unique_classes, counts = np.unique(var_values, return_counts=True)
            target_distribution = {cls: batch_size // len(unique_classes) for cls in unique_classes}

        self.target_distribution = target_distribution

        # Create a dictionary to store indices for each class
        self.class_indices = {label: [] for label in set(var_values)}
        for idx, label in enumerate(var_values):
            self.class_indices[label].append(idx)
        
        print(f"Target distribution: {self.target_distribution}")

    def __iter__(self):
        # Shuffle indices for each class
        for label in self.class_indices:
            random.shuffle(self.class_indices[label])
        
        # Generate balanced batches
        batch = []
        class_counts = {label: 0 for label in self.target_distribution.keys()}
        while len(batch) < self.num_samples:
            for label in self.target_distribution:
                for _ in range(self.target_distribution[label]):
                    if class_counts[label] < len(self.class_indices[label]):
                        batch.append(self.class_indices[label][class_counts[label]])
                        class_counts[label] += 1
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
    
    def __len__(self):
        return self.num_samples // self.batch_size


# class BalancedBatchSampler(Sampler):
#     def __init__(self, data_source, labels, batch_size, target_variable, target_distribution=None):
#         self.data_source = data_source
#         self.labels = [label[target_variable] for label in labels]  # Extract the target variable from each label
#         self.batch_size = batch_size
#         self.num_samples = len(data_source)

#         # If target distribution is not provided, create a uniform distribution
#         if target_distribution is None:
#             unique_classes, counts = np.unique(self.labels, return_counts=True)
#             target_distribution = {cls: batch_size // len(unique_classes) for cls in unique_classes}

#         self.target_distribution = target_distribution

#         # Create a dictionary to store indices for each class
#         self.class_indices = {label: [] for label in set(self.labels)}
#         for idx, label in enumerate(self.labels):
#             self.class_indices[label].append(idx)
        
#         print(f"Target distribution: {self.target_distribution}")

#     def __iter__(self):
#         # Shuffle indices for each class
#         for label in self.class_indices:
#             random.shuffle(self.class_indices[label])
        
#         # Generate balanced batches
#         batch = []
#         class_counts = {label: 0 for label in self.target_distribution.keys()}
#         while len(batch) < self.num_samples:
#             for label in self.target_distribution:
#                 for _ in range(self.target_distribution[label]):
#                     if class_counts[label] < len(self.class_indices[label]):
#                         batch.append(self.class_indices[label][class_counts[label]])
#                         class_counts[label] += 1
#             if len(batch) >= self.batch_size:
#                 yield batch[:self.batch_size]
#                 batch = batch[self.batch_size:]
    
#     def __len__(self):
#         return self.num_samples // self.batch_size


# class BalancedBatchSampler(Sampler):

#     def __init__(self, data_source, labels, batch_size, target_distribution):
#         self.data_source = data_source
#         self.labels = labels
#         self.batch_size = batch_size
#         self.target_distribution = target_distribution
#         self.num_samples = len(data_source)
        
#         # Create a dictionary to store indices for each class
#         self.class_indices = {label: [] for label in set(labels)}
#         for idx, label in enumerate(labels):
#             self.class_indices[label].append(idx)
        
#     def __iter__(self):
#         # Shuffle indices for each class
#         for label in self.class_indices:
#             random.shuffle(self.class_indices[label])
        
#         # Generate balanced batches
#         batch = []
#         class_counts = {label: 0 for label in self.target_distribution.keys()}
#         while len(batch) < self.num_samples:
#             for label in self.target_distribution:
#                 for _ in range(self.target_distribution[label]):
#                     if class_counts[label] < len(self.class_indices[label]):
#                         batch.append(self.class_indices[label][class_counts[label]])
#                         class_counts[label] += 1
#             if len(batch) >= self.batch_size:
#                 yield batch[:self.batch_size]
#                 batch = batch[self.batch_size:]
    
#     def __len__(self):
#         return self.num_samples // self.batch_size
