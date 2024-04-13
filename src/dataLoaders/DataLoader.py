from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import numpy as np
import os

class DataSetLoader:
    def __init__(self, data_dir, image_size=(256, 256), batch_size=32 ):
        self.data_dir = data_dir
        self.image_size = image_size
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
                if filename.endswith('.png'):
                    img_path = os.path.join(self.data_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # print(img)
                    # plot image
                    # plt.imshow(img)
                    img = cv2.resize(img, self.image_size)
                    label = int(filename.split('_')[-1].split('.')[0])
                    images.append(img)
                    labels.append(label)

            # Normalize and add channel dimension
            images = np.array(images, dtype=np.float32) / 255.0
            images = np.expand_dims(images, axis=1)  # Shape: [num_images, 1, height, width]
            labels = np.array(labels, dtype=np.float32)

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