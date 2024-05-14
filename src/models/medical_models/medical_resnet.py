# from monai.networks.nets import ResNet, resnet18, resnet34, resnet50
import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torchvision.models import ResNet, resnet18, resnet34, resnet50
from torchvision import models
from src.models.medical_models.base_medical import MedicalResNetModelBase


class MedicalResNetModel(MedicalResNetModelBase):
    def __init__(self, num_epochs, data_loader: NiftiDataLoader, learning_rate=0.01, weight_decay=None, dropout_rate=None, depth=18):
        super().__init__(num_epochs, data_loader, learning_rate, weight_decay, dropout_rate, depth, pretrained=False)

    def set_model(self):
        if self.depth == 18:
            self.model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.depth == 34:
            self.model = resnet34(weights=models.ResNet18_Weights.DEFAULT)
        elif self.depth == 50:
            self.model = resnet50(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported depth for ResNet. Choose from 18, 34, 50.")
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        out_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(self.n_input_channels, out_channels, kernel_size=self.model.conv1.kernel_size, 
                                     stride=self.model.conv1.stride, padding=self.model.conv1.padding, 
                                     bias=False)
        

