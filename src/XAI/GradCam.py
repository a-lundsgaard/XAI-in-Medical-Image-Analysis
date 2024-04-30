import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
# from src.XAI.utils.save_plots import save_saliency_maps
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset
from src.XAI.utils.BaseXAI import BaseXAI


class GradCamResnet(BaseXAI):
    def __init__(self, modelWrapper: ResNetModel, layerCount=20):
        super().__init__(modelWrapper)
        self.layerCount = layerCount
    
    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir=None, externalEvalData: TensorDataset = None, plot=True):
        
        input_image, input_label = self.get_image_and_label(index, use_test_data, externalEvalData)

        target_layer = self.find_last_conv_layer(self.modelWrapper.model)
        # Hook the target layer to access its feature maps and gradients
        features = []
        grads = []

        def hook_features(module, input, output):
            # print("Feature map mean:", output.mean().item())
            features.append(output)

        def hook_grads(module, grad_in, grad_out):
            # print("Gradient mean:", grad_out[0].mean().item())
            grads.append(grad_out[0])

        hook_f = target_layer.register_forward_hook(hook_features)
        hook_b = target_layer.register_backward_hook(hook_grads)

        # Forward pass
        output: Tensor = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        # Backward pass
        if self.modelWrapper.model.fc.out_features == 1:  # Regression
            target: Tensor = output
        else:  # Classification
            target = output.argmax(dim=1)
        target.backward()

        # Remove hooks
        hook_f.remove()
        hook_b.remove()

        # Process the feature maps and gradients
        gradients = grads[0]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        feature_maps = features[0]
        for i in range(feature_maps.shape[1]):
            feature_maps[:, i, :, :] *= pooled_gradients[i]

        # Generate the heatmap
        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        max_val = torch.max(heatmap)

        if max_val > 0:
            heatmap = F.relu(heatmap)
            heatmap /= max_val
        else:
            min_val = torch.min(heatmap).abs()
            print("Min value in heatmap before ReLU is:", min_val)
            heatmap += min_val*0.01
            heatmap /= torch.max(heatmap)
            print("Max value in heatmap after ReLU is zero3.")

        # Resize heatmap to the input image size
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(input_image.size(2), input_image.size(3)),
            mode='bilinear',
            align_corners=False
        ).squeeze()


        # Convert heatmap to numpy for visualization
        self.heatmap = heatmap
        heatmap_np = heatmap.cpu().detach().numpy()

        # Visualization
        if plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(input_image.cpu().squeeze(), cmap='gray')
            plt.title("Input Image")
            plt.title(f"Input Image (Label: {input_label})")

            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(heatmap_np, cmap='jet')
            # plt.title("Grad-CAM")
            plt.title(f"Grad-CAM (Prediction: {round(target.item(), 2)})")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            img_np = input_image.cpu().squeeze().numpy()
            plt.imshow(img_np, cmap='gray', interpolation='nearest')
            plt.imshow(heatmap_np, cmap='jet', alpha=0.5, interpolation='nearest')
            plt.title("Overlay")
            plt.axis('off')
            
            if save_dir and save_output:
                self.fileSaver.set_custom_save_dir(save_dir, save_output)
                self.fileSaver.handleSaveImage(index, plt, f"grad_cam_{input_label}")
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

