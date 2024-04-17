import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset

class ModifiedGradCam:
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper
        self.fileSaver = PLTSaver(self.__class__.__name__)
        self.heatmap: Tensor = None

    def __find_last_conv_layer(self, model: torch.nn.Module):
        """
        Find the last convolutional layer in the model for Grad-CAM.
        """
        conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module
        return conv_layer

    def generate_grad_cam(self, index=0, true_value=None):
        """
        Generate Grad-CAM visualization for a given input image index from the test dataset.
        Args:
            index (int): Index of the test image
            true_value (float): True value of the output variable for regression analysis
        Returns:
            heatmap (Tensor): Computed heatmap for visualization
        """
        input_image, input_label = self.modelWrapper.get_single_test_image(index)
        target_layer = self.__find_last_conv_layer(self.modelWrapper.model)

        features = []
        grads = []

        def hook_features(module, input, output):
            features.append(output)

        def hook_grads(module, grad_in, grad_out):
            grads.append(grad_out[0])

        hook_f = target_layer.register_forward_hook(hook_features)
        hook_b = target_layer.register_backward_hook(hook_grads)

        output = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        pred_value: Tensor = output
        true_value = torch.tensor([input_label], device=pred_value.device)
        # Using the inverse of the absolute difference as a surrogate for 'closeness'
        d = 1 / (torch.abs(pred_value - true_value) + 0.0001)  # Adding a small constant to avoid division by zero
        d.backward()

        hook_f.remove()
        hook_b.remove()

        gradients = grads[0]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).abs()
        feature_maps = features[0]
        # feature_maps = feature_maps.abs()
        for i in range(feature_maps.shape[1]):
            feature_maps[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        #heatmap = F.relu(heatmap)  # Use ReLU to only consider positive influences
        max_val = torch.max(heatmap)
        #heatmap /= torch.max(heatmap)  # Normalize the heatmap
        if max_val > 0:
            heatmap = F.relu(heatmap)
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = torch.abs(heatmap)
            max_val = torch.max(heatmap)
            heatmap -= max_val*0.80
            #print("Min value in heatmap before ReLU is:", min_val)
            #grad_cam_map += min_val
            # heatmap *= -  1
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)

        # Resize heatmap to the input image size
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0),
                                size=(input_image.size(2), input_image.size(3)),
                                mode='bilinear',
                                align_corners=False).squeeze()

        self.heatmap = heatmap
        # Convert heatmap to numpy for visualization
        heatmap_np = heatmap.cpu().detach().numpy()
        img_np = input_image.cpu().squeeze().numpy()

        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title(f"Input Image (Label: {input_label})")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_np, cmap='jet')
        plt.title(f"Grad-CAM (Prediction: {round(pred_value.item(), 2)})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(img_np, cmap='gray', interpolation='nearest')
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5, interpolation='nearest')
        plt.title("Overlay")
        plt.axis('off')

        plt.show()

        # Optionally save the figure using the file saver
        self.fileSaver.handleSaveImage(index, plt, f"grad_cam_{input_label}")

        #return heatmap