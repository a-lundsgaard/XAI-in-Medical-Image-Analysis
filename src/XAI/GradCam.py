import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
# from src.XAI.utils.save_plots import save_saliency_maps
from src.XAI.utils.SaveFiles import PLTSaver


class GradCamResnet:
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper
        self.fileSaver = PLTSaver(self.__class__.__name__)

    def __find_last_conv_layer(self, model: torch.nn.Module):
        """
        Find the last convolutional layer in the model for Grad-CAM.
        """
        conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module
        return conv_layer
    
    def generateMultipleGradCam(self, image_count=1, use_test_data=True, save_output=False, save_dir="default"):
        """
        Generate Grad-CAM visualization for a set of images.
        Args:
            image_count (int): The number of images for which to generate Grad-CAM visualizations.
        """
        self.fileSaver.set_custom_save_dir(save_dir, save_output)

        max_image_count = self.modelWrapper.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        for i in range(count):
            self.__generate_grad_cam(index=i, use_test_data=use_test_data)

    def __generate_grad_cam(self, index=0, use_test_data=True):
        """
        Generate Grad-CAM visualization for a given input image index from the test dataset.
        """
        if use_test_data:
            input_image, input_label = self.modelWrapper.get_single_test_image(index)
        else:
            input_image, input_label = self.modelWrapper.get_single_train_image(index)

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

        self.fileSaver.handleSaveImage(index, plt, f"grad_cam_{input_label}")

        plt.show()
