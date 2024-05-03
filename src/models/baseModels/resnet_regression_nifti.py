import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torch import Tensor
from torcheval.metrics import R2Score

# Assuming NiftiDataLoader is properly defined elsewhere
from src.dataLoaders.NiftiDataLoader import NiftiDataLoader
from src.dataLoaders.DataLoader import DataSetLoader

class ResNetModel:
    def __init__(self, num_epochs, learning_rate, weight_decay, early_stopping_tol, early_stopping_min_delta, 
                 image_size=(256, 256), batch_size=32, depth=18, data_dir = '../../artificial_data/noisy_generated_images',
                 dataLoader: NiftiDataLoader | DataSetLoader = None,
                 dropout_rate=None,
                 ):
        self.num_epochs = num_epochs
        
        self.dataLoader = dataLoader or DataSetLoader(data_dir, image_size, batch_size) 
        
        self.in_channels = 1 if isinstance(dataLoader, DataLoader) else 3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.early_stopping_tol = early_stopping_tol
        self.early_stopping_min_delta = early_stopping_min_delta

        # Choose the appropriate ResNet architecture
        if depth == 18:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif depth == 34:
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif depth == 50:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported depth for ResNet. Choose from 18, 34, 50.")
        
        # Adapt the first convolution layer to accept 3 channels if using NiftiDataLoader
        out_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(self.in_channels, out_channels, kernel_size=self.model.conv1.kernel_size, 
                                     stride=self.model.conv1.stride, padding=self.model.conv1.padding, 
                                     bias=False)
        
        # Modify the fully connected layer
        num_features = self.model.fc.in_features
        if dropout_rate:
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 1)
            )
        else:
            self.model.fc = nn.Linear(num_features, 1)  # Assuming a regression task

        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Rest of the class remains unchanged...

