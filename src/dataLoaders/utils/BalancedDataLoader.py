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

        # print(f"Var values: {var_values}")

        # If target distribution is not provided, create a uniform distribution
        if target_distribution is None:
            unique_classes, counts = np.unique(var_values, return_counts=True)
            # Ensure at least 1 sample per class if batch_size is smaller than the number of classes
            target_distribution = {cls: max(1, batch_size // len(unique_classes)) for cls in unique_classes}

        self.target_distribution = target_distribution

        # print("Unique classes: ", unique_classes)
        print(f"Target distribution: {self.target_distribution}")

        # Create a dictionary to store indices for each class
        self.class_indices = {label: [] for label in set(var_values)}
        for idx, value in enumerate(var_values):
            self.class_indices[value].append(idx)
        
        # print(f"Target distribution: {self.target_distribution}")
        # print(f"Class indices: {self.class_indices}")

    def __iter__(self):
        # Shuffle indices for each class
        for label in self.class_indices:
            random.shuffle(self.class_indices[label])

        # Generate balanced batches
        batch = []
        class_counts = {label: 0 for label in self.target_distribution.keys()}
        # print(f"Class counts: {class_counts}")
        batches_created = 0

        used_indices = set()

        while batches_created < (self.num_samples // self.batch_size):
            for label in self.target_distribution:
                for _ in range(self.target_distribution[label]):
                    if class_counts[label] < len(self.class_indices[label]):
                        index = self.class_indices[label][class_counts[label]]
                        if index < self.num_samples and index not in used_indices:  # Ensure index is within bounds and not used
                            batch.append(index)
                            used_indices.add(index)
                            class_counts[label] += 1
                            if len(batch) == self.batch_size:
                                # print(f"Batch: {batch}")
                                # print(f"Type of batch: {type(batch)}")
                                yield batch
                                batch = []
                                batches_created += 1
                                break
            if len(batch) > 0 and len(batch) < self.batch_size:
                random.shuffle(batch)
                # print(f"Type of batch: {type(batch)}")
                # print(f"Batch: {batch}")9
                yield batch
                batch = []
                batches_created += 1

        # print(f"Batches created: {batches_created}")

    def __len__(self):
        return self.num_samples // self.batch_size
