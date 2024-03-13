import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

class ResNetModel:
    def __init__(self, data_dir = '../../artificial_data/noisy_generated_images', image_size=(256, 256), num_classes=5, batch_size=32, num_epochs=5):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # include the data in the model 
        self.testData = None

        
        # Initialize the model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1) # Output layer for regression.
        self.model.to(self.device) 

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def load_data(self):
        images = []
        labels = []

        for filename in os.listdir(self.data_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(self.data_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_size)
                label = int(filename.split('_')[-1].split('.')[0])
                images.append(img)
                labels.append(label)

        # Convert to tensors and create dataloaders
        images = np.array(images).transpose((0, 3, 1, 2)) / 255.0
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
        self.testData = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_single_test_image(self, index=0):
        # Assuming self.testData is a TensorDataset
        if self.testData is not None:
            # Fetch the image and label tensors
            img_tensor, label_tensor = self.testData[index]
            # Add a batch dimension, convert to float, and move to the correct device
            img_tensor = img_tensor.unsqueeze(0).float().to(self.device)
            # Move the label to the correct device
            label_tensor = label_tensor.to(self.device)
            return img_tensor, label_tensor.item()  # Return the image tensor and the label as an integer
        else:
            print("Test data not loaded.")
            return None, None


    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad() # Zero the parameter gradients
                outputs = self.model(images).flatten()
                loss = self.criterion(outputs, labels.float()) 
                loss.backward() # TODO: Convert labels to float.
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(self.train_loader)}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.data # TODO: fixx evaluation.



# Example usage
# data_dir = '../../artificial_data/noisy_generated_images'
# model = ResNetModel()
# model.load_data()
# model.train()
# model.evaluate()
