import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
# Assuming ResNetModel is correctly imported from your project structure
from src.models.baseModels.resnet_regression import ResNetModel
from torch import Tensor
import os
import shutil


class XAIResNet:
    def __init__(self, modelWrapper: ResNetModel, device):
        self.modelWrapper = modelWrapper
        self.device = device

    def get_saliency_maps(self, image_count=1, save_output=False, save_dir="default"):
        """
        Generate saliency maps for a set of images.
        Args:
            image_count (int): The number of images for which to generate saliency maps.
        """
        data_dir = f"outputs/saliency_maps/{save_dir}"
        if save_output:
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            os.makedirs(data_dir)

        max_image_count = self.modelWrapper.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        for i in range(count):
            input_image, input_label = self.modelWrapper.get_single_test_image(index=i)
            self.__generate_saliency_map(input_image, input_label, i=i, save_output=save_output, data_dir=data_dir)
    
    def __generate_saliency_map(self, input_image: Tensor, input_label, i=0, save_output=False, data_dir=""):
        """
        Generate a saliency map for a given input image and label.
        Args:
            input_image (Tensor): The input image tensor.
            input_label (int): The label of the input image.
        """
        input_image = input_image.to(self.device).requires_grad_(True)
        
        # Forward pass
        output = self.modelWrapper.model(input_image)  # input_image should already have batch dimension
        self.modelWrapper.model.zero_grad() # Clear any previously stored gradients
        
        # Target for backprop
        target: Tensor = output[0]
        target.backward()
        
        # Saliency map
        saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)  # Take the max across the channels
        saliency = saliency.reshape(self.modelWrapper.image_size)  # Assuming modelWrapper has attribute image_size

        # For grayscale images, the original image is 2D, so no need for color channel conversion
        original_image = input_image.detach().cpu().squeeze().numpy()  # Remove batch dimension and convert to numpy

        # Plotting both original image and saliency map side by side
        plt.figure(figsize=(10, 5))  # Set figure size

        # Plot original image
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.imshow(original_image, cmap='gray')
        # make the title with the predicted label
        plt.title(f"Original Image (Label: {input_label})")
        plt.axis('off')
        
        n_rows = 1
        n_cols = 2
        # Plot saliency map
        plt.subplot(n_rows, n_cols, 2)  # second subplot
        plt.imshow(saliency.cpu(), cmap='hot')
        plt.title(f"Saliency Map (Prediction: {round(target.item(), 2)})")
        plt.axis('off')

        # Save the saliency map
        if save_output:
            plt.savefig(f"{data_dir}/{i}_saliency_map_{input_label}.png")

        plt.show()

    
    
    def get_grad_cam(self, image_count=1, save_output=False, save_dir="default"):
        """
        Generate Grad-CAM visualizations for a set of images.
        """
        data_dir = f"outputs/grad_cam/{save_dir}"
        if save_output:
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            os.makedirs(data_dir)
        
        max_image_count = self.modelWrapper.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        for i in range(count):
            input_image, input_label = self.modelWrapper.get_single_test_image(index=i)
            self.__generate_grad_cam(input_image, input_label, i=i, save_output=save_output, data_dir=data_dir)
    
    def __generate_grad_cam(self, input_image: Tensor, input_label, i=0, save_output=False, data_dir=""):
        """
        Generate a Grad-CAM visualization for a given input image and label.
        """
        # Set model to evaluation mode
        self.modelWrapper.model.eval()
        
        # Hook the feature extractor
        features = []
        grads = []
        def hook_function(module, input, output):
            features.append(output)
            return None

        def hook_function_backward(module, grad_in, grad_out):
            grads.append(grad_out[0])
            return None
        
        final_layer = self.modelWrapper.model.layer4[-1]  # This needs to be adapted to your model architecture
        final_layer.register_forward_hook(hook_function)
        final_layer.register_backward_hook(hook_function_backward)

        input_image = input_image.unsqueeze(0).to(self.device)  # Add batch dimension and send to device
        input_image.requires_grad = True
        
        # Forward pass
        outputs = self.modelWrapper.model(input_image)
        score, preds = torch.max(outputs, 1)  # Get the highest scoring class
        target_class = preds[0]
        
        # Backward pass
        self.modelWrapper.model.zero_grad()
        score.backward()

        # Gradient and feature map
        gradients = grads[0].detach()
        feature_map = features[0].detach()

        # Weight the channels by the gradients
        pooled_gradients = torch.mean(gradients, [0, 2, 3])
        for j in range(feature_map.shape[1]):  # Number of channels
            feature_map[:, j, :, :] *= pooled_gradients[j]
        
        # Average the channels of the feature maps
        grad_cam = torch.mean(feature_map, 1).squeeze()
        
        # Apply ReLU
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam - grad_cam.min()
        grad_cam = grad_cam / grad_cam.max()
        
        grad_cam = grad_cam.cpu().numpy()
        
        # Resize Grad-CAM map to the size of the input image
        grad_cam = np.uint8(255 * grad_cam)  # Convert to uint8
        grad_cam = np.resize(grad_cam, (input_image.shape[2], input_image.shape[3]))  # Assuming input_image is CxHxW
        
        # Visualization and Saving Logic Here (Similar to __generate_saliency_map)
