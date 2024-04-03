import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel

class GradCamResnet:
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper
        # self.device = device

    def __find_last_conv_layer(self, model: torch.nn.Module):
        """
        Find the last convolutional layer in the model for Grad-CAM.
        """
        conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module
        return conv_layer

    def generate_grad_cam(self, index=0, use_test_data=True):
        """
        Generate Grad-CAM visualization for a given input image index from the test dataset.
        """
        if use_test_data:
            input_image, _ = self.modelWrapper.get_single_test_image(index)
        else:
            input_image, _ = self.modelWrapper.get_single_train_image(index)

        # Find the last convolutional layer
        target_layer = self.__find_last_conv_layer(self.modelWrapper.model)
        if target_layer is None:
            print("No convolutional layer found.")
            return

        # Hook the target layer to access its feature maps and gradients
        features = []
        grads = []

        def hook_features(module, input, output):
            features.append(output)

        def hook_grads(module, grad_in, grad_out):
            grads.append(grad_out[0])

        hook_f = target_layer.register_forward_hook(hook_features)
        hook_b = target_layer.register_backward_hook(hook_grads)

        # Forward pass
        output = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        # Backward pass
        if self.modelWrapper.model.fc.out_features == 1:  # Regression
            target = output
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
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)  # Normalize the heatmap

        # Resize heatmap to the input image size
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(input_image.size(2), input_image.size(3)),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Convert heatmap to numpy for visualization
        heatmap_np = heatmap.cpu().detach().numpy()

        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(input_image.cpu().squeeze(), cmap='gray')
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_np, cmap='jet')
        plt.title("Grad-CAM")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        img_np = input_image.cpu().squeeze().numpy()
        plt.imshow(img_np, cmap='gray', interpolation='nearest')
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5, interpolation='nearest')
        plt.title("Overlay")
        plt.axis('off')

        plt.show()
