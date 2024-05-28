# from monai.networks.nets import ResNet, resnet18, resnet34, resnet50
import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torchvision.models import ResNet, resnet18, resnet34, resnet50
from torchvision import models
from src.models.medical_models.base_medical import MedicalResNetModelBase
from src.models.medical_models.utils.AdaptiveSlicing import AdaptiveSlicingModule

class AdaptiveMedicalResNetModel(nn.Module, MedicalResNetModelBase):
    def __init__(self, num_epochs, data_loader: NiftiDataLoader, learning_rate=0.01, weight_decay=None, dropout_rate=None, depth=18):
        super().__init__(num_epochs, data_loader, learning_rate, weight_decay, dropout_rate, depth, pretrained=False)
        self.adaptive_slicing = AdaptiveSlicingModule()

    def set_model(self):
        if self.depth == 18:
            self.model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.depth == 34:
            self.model = resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif self.depth == 50:
            self.model = resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported depth for ResNet. Choose from 18, 34, 50.")
        
        out_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(9, out_channels, kernel_size=self.model.conv1.kernel_size, 
                                     stride=self.model.conv1.stride, padding=self.model.conv1.padding, 
                                     bias=False)
        self.spatial_dims = 2

        self.model_forward = self.model.forward
        self.model.forward = self.forward

    def forward(self, x):
        # if torch.Size([32, 9, 128, 128]) then don't do anything
        # if torch.Size([32, 1, 128, 128, 128]) then do the following
        # print("x size forward", x.size())
        if x.size(1) == 1:
            x = self.adaptive_slicing(x)

        # print("AdaptiveMedicalResNetModel forward", x.shape)
        # # Apply adaptive slicing
        # # x = self.adaptive_slicing(x)
        # print("AdaptiveMedicalResNetModel forward after adaptive slicing", x.shape)
        # Ensure the tensor shape is (N, C, H, W)
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.view(N, C, H, W)
        return self.model(x)
