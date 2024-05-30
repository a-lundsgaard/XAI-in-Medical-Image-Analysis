import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.medical_models.combined_medical import MedicalCombinedResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset
from src.XAI.utils.BaseXAI import BaseXAI

class GradCamResnet(BaseXAI):
    def __init__(self, modelWrapper: MedicalCombinedResNetModel, layerCount=20):
        super().__init__(modelWrapper)
        self.layerCount = layerCount
        self.features = {'axial': None, 'coronal': None, 'sagittal': None}
        self.gradients = {'axial': None, 'coronal': None, 'sagittal': None}
        self.modelWrapper.model.eval()

    def save_features_axial(self, module, input, output):
        self.features['axial'] = output

    def save_gradients_axial(self, module, grad_in, grad_out):
        self.gradients['axial'] = grad_out[0]

    def save_features_coronal(self, module, input, output):
        self.features['coronal'] = output

    def save_gradients_coronal(self, module, grad_in, grad_out):
        self.gradients['coronal'] = grad_out[0]

    def save_features_sagittal(self, module, input, output):
        self.features['sagittal'] = output

    def save_gradients_sagittal(self, module, grad_in, grad_out):
        self.gradients['sagittal'] = grad_out[0]

    def register_hooks(self):
        self.modelWrapper.model.axial_features[-1].register_forward_hook(self.save_features_axial)
        self.modelWrapper.model.axial_features[-1].register_backward_hook(self.save_gradients_axial)
        self.modelWrapper.model.coronal_features[-1].register_forward_hook(self.save_features_coronal)
        self.modelWrapper.model.coronal_features[-1].register_backward_hook(self.save_gradients_coronal)
        self.modelWrapper.model.sagittal_features[-1].register_forward_hook(self.save_features_sagittal)
        self.modelWrapper.model.sagittal_features[-1].register_backward_hook(self.save_gradients_sagittal)

    def generate_heatmap(self, features, gradients, original_size):
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(features.shape[1]):
            features[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(features, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # Resize heatmap to the original input image size
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=original_size, mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        return heatmap

    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir=None, externalEvalData: TensorDataset = None, plot=True):
        input_image, input_label = self.get_image_and_label(index, use_test_data, externalEvalData)
        input_image = input_image.unsqueeze(0).to(self.device)  # Add batch dimension and move to correct device

        original_size = (input_image.size(2), input_image.size(3))  # Get the original input image size

        # Register hooks
        self.register_hooks()

        # Forward pass
        output = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        # Backward pass
        target = output
        target.backward()

        # Generate heatmaps for each view
        heatmaps = {
            'axial': self.generate_heatmap(self.features['axial'], self.gradients['axial'], original_size),
            'coronal': self.generate_heatmap(self.features['coronal'], self.gradients['coronal'], original_size),
            'sagittal': self.generate_heatmap(self.features['sagittal'], self.gradients['sagittal'], original_size),
        }

        # Visualization
        if plot:
            fig, axes = plt.subplots(3, 4, figsize=(24, 12))
            view_titles = ['Axial View', 'Coronal View', 'Sagittal View']

            for i, view in enumerate(['axial', 'coronal', 'sagittal']):
                view_img = input_image.cpu().squeeze()[i * self.modelWrapper.model.n_input_channels:(i + 1) * self.modelWrapper.model.n_input_channels]
                for j in range(self.modelWrapper.model.n_input_channels):
                    axes[i, j].imshow(view_img[j], cmap='gray')
                    axes[i, j].imshow(heatmaps[view], cmap='jet', alpha=0.5)
                    axes[i, j].set_title(f"{view_titles[i]} - Channel {j + 1}")
                    axes[i, j].axis('off')

                # Overlay all channels
                combined_image = np.mean(view_img.cpu().detach().numpy(), axis=0)
                axes[i, 3].imshow(combined_image, cmap='gray')
                axes[i, 3].imshow(heatmaps[view], cmap='jet', alpha=0.5)
                axes[i, 3].set_title(f"{view_titles[i]} - Combined Overlay")
                axes[i, 3].axis('off')

            if save_dir and save_output:
                self.fileSaver.set_custom_save_dir(save_dir, save_output)
                self.fileSaver.handleSaveImage(index, plt, f"grad_cam_{input_label}")

            plt.tight_layout()
            plt.show()

    def generateMultipleGradCam(self, image_count=1, use_test_data=True, save_output=False, save_dir="default", externalEvalData: TensorDataset = None):
        """
        Generate Grad-CAM visualization for a set of images.
        Args:
            image_count (int): The number of images for which to generate Grad-CAM visualizations.
        """
        self.fileSaver.set_custom_save_dir(save_dir, save_output)

        max_image_count = self.modelWrapper.dataLoader.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        for i in range(count):
            self.generate_map(index=i, use_test_data=use_test_data, externalEvalData=externalEvalData)
