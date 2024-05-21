import torch
from torch.utils.data import Sampler
import random

class BalancedBatchSampler(Sampler):


    def __init__(self, data_source, labels, batch_size, target_distribution):
        self.data_source = data_source
        self.labels = labels
        self.batch_size = batch_size
        self.target_distribution = target_distribution
        self.num_samples = len(data_source)
        
        # Create a dictionary to store indices for each class
        self.class_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
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
