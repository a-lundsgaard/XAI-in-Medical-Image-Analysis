import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision import models
from torch import Tensor
from torcheval.metrics import R2Score
from src.dataLoaders.DataLoader import DataSetLoader

class ResNetModel:
    def __init__(self, num_epochs, learning_rate, weight_decay, early_stopping_tol, early_stopping_min_delta, 
                 image_size=(256, 256), batch_size=32, depth=18, data_dir = '../../artificial_data/noisy_generated_images',
                 dataLoader: DataSetLoader = None,
                 dropout_rate=None
                 ):
        self.num_epochs = num_epochs

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.dataLoader = dataLoader or DataSetLoader(data_dir, image_size, batch_size)

        self.early_stopping_tol = early_stopping_tol
        self.early_stopping_min_delta = early_stopping_min_delta

        if depth == 18:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif depth == 34:
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif depth == 50:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported depth for ResNet. Choose from 18, 34, 50.")
            
        out_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(1, out_channels, kernel_size=self.model.conv1.kernel_size, 
                                     stride=self.model.conv1.stride, padding=self.model.conv1.padding, 
                                     bias=False)
            
        num_features = self.model.fc.in_features

        if dropout_rate:
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 1)
            )
        else:
            self.model.fc = nn.Linear(num_features, 1)  # Output layer for regression.

        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def get_single_image(self, data: TensorDataset, index=0):
        # Assuming self.testData is a TensorDataset
        if data is not None:
            # Fetch the image and label tensors
            if index < len(data):
                img_tensor, label_tensor = data[index]
                # Add a batch dimension, convert to float, and move to the correct device
                img_tensor = img_tensor.unsqueeze(0).float().to(self.device)
                # Move the label to the correct device
                label_tensor = label_tensor.to(self.device)
                return img_tensor, label_tensor.item()
        else:
            print(f"Data is not loaded")
            return None, None

    def get_single_test_image(self, index=0):
        return self.get_single_image(self.dataLoader.testData, index)
    
    def get_single_train_image(self, index=0):
        return self.get_single_image(self.dataLoader.trainingData, index)

    def load_data(self):
        self.dataLoader.load_data()

    def train(self):
        print("Is cuda available: ", torch.cuda.is_available())
        print("Is mps available: ", torch.backends.mps.is_available())
        early_stopping = EarlyStopping(tolerance=self.early_stopping_tol, min_delta=self.early_stopping_min_delta)
        epoch_val_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            running_train_loss = 0.0
            running_val_loss = 0.0
            
            for images, labels in self.dataLoader.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images).flatten()
                loss: Tensor = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()
                running_val_loss += self.validation_loss()
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {running_train_loss/len(self.dataLoader.train_loader)}, Val Loss: {running_val_loss/len(self.dataLoader.val_loader)}")

            epoch_val_losses.append(running_val_loss/len(self.dataLoader.val_loader))
                            
            # Early stopping
            if epoch > 2:
                early_stopping(epoch_val_losses)
                if early_stopping.early_stop:
                    print("Early stopped at epoch:", epoch + 1)
                    break

    def validation_loss(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.dataLoader.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).flatten()
                loss = self.criterion(outputs, labels.float()) 
                return loss.item()

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        # Initialize the R2Score metric and move it to the same device as model and data
        r2_metric = R2Score().to(self.device)
        with torch.no_grad():
            for images, labels in self.dataLoader.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predicted = outputs.flatten()

                loss = self.criterion(predicted, labels)
                running_loss += loss.item()
                r2_metric.update(predicted, labels)

        # Compute final R2 score and average loss
        r2_score = r2_metric.compute()
        average_loss = running_loss / len(self.dataLoader.test_loader)
        print(f'Loss of the network on the test images: {average_loss}')
        print(f'R^2 score of the network on the test images: {r2_score}')

class EarlyStopping:
    def __init__(self, tolerance, min_delta):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch_val_losses):
        if (epoch_val_losses[-1] - epoch_val_losses[-2]) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


# Example usage
# data_dir = '../../artificial_data/noisy_generated_images'
# model = ResNetModel()
# model.load_data()
# model.train()
# model.evaluate()
