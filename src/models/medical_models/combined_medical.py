import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import models
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from src.models.medical_models.base_medical import MedicalResNetModelBase

class CombinedResNetModel(nn.Module):
    def __init__(self, n_input_channels=1, resnet_depth=18, num_labels=1, pretrained=True, only_center_slice=False):
        super(CombinedResNetModel, self).__init__()
        
        # Initialize ResNet models for each view
        self.resnet_axial = self._initialize_resnet(resnet_depth, n_input_channels, pretrained)
        self.resnet_coronal = self._initialize_resnet(resnet_depth, n_input_channels, pretrained)
        self.resnet_sagittal = self._initialize_resnet(resnet_depth, n_input_channels, pretrained)
        self.resnet_depth = resnet_depth
        self.n_input_channels = n_input_channels
        self.only_center_slice = only_center_slice
        
        # Feature extractor layers
        self.axial_features = nn.Sequential(*list(self.resnet_axial.children())[:-1])
        self.coronal_features = nn.Sequential(*list(self.resnet_coronal.children())[:-1])
        self.sagittal_features = nn.Sequential(*list(self.resnet_sagittal.children())[:-1])
        
        # Fully connected layer for combining features
        num_features = self.resnet_axial.fc.in_features * 3  # multiply by 3 since we combine three views
        self.fc = nn.Linear(num_features, num_labels)
        
    def _initialize_resnet(self, depth, n_input_channels, pretrained):
        model = None
        if depth == 18:
            w18 = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=w18)
        elif depth == 34:
            w34 = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=w34)
        elif depth == 50:
            w50 = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=w50)
        else:
            raise ValueError("Unsupported depth for ResNet. Choose from 18, 34, 50.")
        
        # Modify the first convolutional layer to match the number of input channels
        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(n_input_channels, out_channels, kernel_size=model.conv1.kernel_size, 
                                stride=model.conv1.stride, padding=model.conv1.padding, 
                                bias=False)
        
        return model

    def forward(self, x):

        if self.only_center_slice and self.n_input_channels == 3:
            # Extract only the center slice for each view
            print("Extracting only the center slice for each view")
            axial_input = x[:, self.n_input_channels-1:self.n_input_channels, :, :]
            coronal_input = x[:, 2*self.n_input_channels-1:2*self.n_input_channels, :, :]
            sagittal_input = x[:, 3*self.n_input_channels-1:3*self.n_input_channels, :, :]
        else:
            axial_input = x[:, :self.n_input_channels, :, :]
            coronal_input = x[:, self.n_input_channels:2*self.n_input_channels, :, :]
            sagittal_input = x[:, 2*self.n_input_channels:, :, :]
        
        # Extract features from each view
        axial_features = self.axial_features(axial_input).view(x.size(0), -1)
        coronal_features = self.coronal_features(coronal_input).view(x.size(0), -1)
        sagittal_features = self.sagittal_features(sagittal_input).view(x.size(0), -1)
        
        # Combine features
        combined_features = torch.cat((axial_features, coronal_features, sagittal_features), dim=1)
        
        # Final prediction
        output = self.fc(combined_features)
        return output


class MedicalCombinedResNetModel(MedicalResNetModelBase):
    def __init__(self, num_epochs, data_loader: NiftiDataLoader, learning_rate=0.01, weight_decay=None, dropout_rate=None, depth=18):
        super().__init__(num_epochs, data_loader, learning_rate, weight_decay, dropout_rate, depth, pretrained=False)
    
    def set_model(self):
        num_labels = len(self.data_loader.train_loader.dataset[0]['label'])
        # make sure that the input channels are divisible by 3
        if self.n_input_channels % 3 != 0:
            raise ValueError("Number of input channels should be divisible by 3.")
        input_channels_per_view = self.n_input_channels // 3
        self.model = CombinedResNetModel(n_input_channels=input_channels_per_view, resnet_depth=self.depth, num_labels=num_labels, pretrained=True)


# Example of how to train this model
# Initialize your data loader
# data_loader = NiftiDataLoader(train_data, val_data, test_data)  # Assuming you have these datasets

# # Instantiate and train the model
# combined_model = MedicalCombinedResNetModel(num_epochs=100, data_loader=data_loader)
# combined_model.train()
# combined_model.evaluate()
