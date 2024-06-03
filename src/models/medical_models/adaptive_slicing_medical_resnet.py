import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import models
from src.models.medical_models.utils.AdaptiveSlicing import AdaptiveSlicingModule
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from src.models.medical_models.base_medical import BaseMedical
from torchvision.models.resnet import ResNet, BasicBlock

class ResnetAdaptive(ResNet):
    def __init__(self):
        super(ResnetAdaptive, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.adaptive_slicing = AdaptiveSlicingModule()
        print("ResnetAdaptive deviation", self.adaptive_slicing.deviation.item())

    def forward(self, x):
        if x.size(1) == 1:
            x = self.adaptive_slicing(x)
        return super(ResnetAdaptive, self).forward(x)
    

class AdaptiveMedicalResNetModel(BaseMedical):
    def __init__(self, num_epochs, data_loader: NiftiDataLoader, learning_rate=0.01, weight_decay=None, dropout_rate=None, depth=18):
        super().__init__(num_epochs, data_loader, learning_rate, weight_decay, dropout_rate, depth, pretrained=False)

    def set_model(self):
        self.model = ResnetAdaptive()
        out_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(9, out_channels, kernel_size=self.model.conv1.kernel_size, 
                                     stride=self.model.conv1.stride, padding=self.model.conv1.padding, 
                                     bias=False)