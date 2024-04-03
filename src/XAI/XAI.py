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
        self.modelWrapper.model.zero_grad()
        
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
        # plt.title(f"Original Image: { }")
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
