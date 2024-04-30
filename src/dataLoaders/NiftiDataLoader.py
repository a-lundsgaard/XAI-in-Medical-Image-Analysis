import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import nibabel as nib
from sklearn.model_selection import train_test_split

class NiftiDataLoader:
    def __init__(self, data_dir, image_size=(256, 256), batch_size=32):
        self.data_dir = data_dir
        self.image_size = image_size  # This can be used if resizing is needed
        self.batch_size = batch_size

        self.trainingData = None
        self.testData = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self):
        images = []
        labels = []

        for filename in os.listdir(self.data_dir):
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                img_path = os.path.join(self.data_dir, filename)
                img = nib.load(img_path).get_fdata()
                img = self.preprocess_image(img)
                img_slices = self.extract_slices(img)
                label = self.extract_label(filename)
                if len(img_slices) == 3:  # Ensuring there are exactly three slices
                    # Stack slices along a new dimension to form multi-channel input
                    multi_channel_image = np.stack(img_slices, axis=0)
                    images.append(multi_channel_image)
                    labels.append(label)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        self.trainingData = train_dataset
        self.testData = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def preprocess_image(self, img):
        # Normalization (you might want to resize or apply other preprocessing)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def extract_slices(self, img):
        c, h, w = img.shape
        # Extract middle slices from each axis
        axial = img[c//2, :, :]
        coronal = img[:, h//2, :]
        sagittal = img[:, :, w//2]
        return [axial, coronal, sagittal]

    def extract_label(self, filename):
        # Example label extraction, assuming label is part of the filename
        return int(filename.split('_')[-1].split('.')[0])

