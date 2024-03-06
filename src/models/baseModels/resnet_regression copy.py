import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

# Parameters
image_size = (256, 256)
num_classes = 5  # Change this if you have a different number of labels
batch_size = 32

# Data Preparation
# data_dir = 'noisy_generated_images'  # Update this to your dataset folder
# data_dir = '../../noisy_generated_images'

data_dir = '../../artificial_data'
path = os.path.join(data_dir, 'noisy_generated_images')
images = []
labels = []

for filename in os.listdir(path):
    if filename.endswith('.png'):
        path = os.path.join(path, filename)
        print(path)

        img = cv2.imread(os.path.join(path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = cv2.resize(img, image_size)
        label = int(filename.split('_')[-1].split('.')[0])  # Extract label from filename
        images.append(img)
        labels.append(label)

# Adjust labels to start from 0
# label_offset = min(labels)
# labels = [label - label_offset for label in labels]  # Make labels start from 0

# print(labels)

# After collecting all labels
label_set = set(labels)  # Find unique label values
label_map = {val: i for i, val in enumerate(sorted(label_set))}  # Create a mapping from original to [0, num_classes-1]
labels = [label_map[val] for val in labels]  # Remap all labels to [0, num_classes-1]

# Now labels are properly mapped, print the unique labels to check
print("Unique remapped labels:", set(labels))

# Convert to PyTorch tensors
images = np.array(images).transpose((0, 3, 1, 2)) / 255.0  # Normalize and rearrange for PyTorch
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust for the number of classes

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 5  # You can change this
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')
