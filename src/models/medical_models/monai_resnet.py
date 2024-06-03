from monai.networks.nets import ResNet, resnet18, resnet34, resnet50
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from src.models.medical_models.base_medical import BaseMedical
import torch


class MonaiMedicalResNet(BaseMedical):
    def __init__(self, num_epochs, data_loader: NiftiDataLoader, learning_rate=0.01, weight_decay=None, dropout_rate=None, depth=18, pretrained=True):
        super().__init__(num_epochs, data_loader, learning_rate, weight_decay, dropout_rate, depth, pretrained)

    def set_model(self):
        resnet: ResNet = None
        if self.depth == 18:
            resnet = resnet18
        elif self.depth == 34:
            resnet = resnet34
        elif self.depth == 50:
            resnet = resnet50
        else:
            raise ValueError(
                "Unsupported depth for ResNet. Choose from 18, 34, 50.")
        
        print("Spatial dims: ", self.spatial_dims)
        self.model: ResNet = resnet(
            spatial_dims=self.spatial_dims,
            n_input_channels=self.n_input_channels,
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
