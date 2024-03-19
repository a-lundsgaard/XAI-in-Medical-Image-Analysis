import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
from torch import Tensor


class XAIResNet:
    def __init__(self, modelWrapper: ResNetModel, device):
        self.modelWrapper = modelWrapper
        self.device = device

    def get_saliency_maps(self, image_count = 1):
        """
        Generate saliency maps for a set of images.
        Args:
            image_count (int): The number of images for which to generate saliency maps.
        """
        # make ternary operator
        max_image_count = self.modelWrapper.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else self.modelWrapper.testData.tensors[0].shape[0]

        for i in range(count):
            input_image, input_label = self.modelWrapper.get_single_test_image(index=i)
            self.__generate_saliency_map(input_image, input_label)
    
    def __generate_saliency_map(self, input_image: Tensor, input_label):
        """
        Generate a saliency map for a given input image and label.
        Args:
            input_image (Tensor): The input image tensor.
            input_label (int): The label of the input image.
        """
        input_image = input_image.to(self.device).requires_grad_(True)
        
        # Forward pass
        output: Tensor = self.modelWrapper.model(input_image)  # input_image should already have batch dimension

        self.modelWrapper.model.zero_grad()
        
        # Target for backprop
        target = output[0]
        
        # Backward pass
        target.backward()
        
        # Saliency map
        saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)  # Take the max across the channels
        saliency = saliency.reshape(self.modelWrapper.image_size)  # Assuming modelWrapper has attribute image_size

        # Original image for plotting (assuming input_image is normalized)
        original_image = input_image.detach().cpu().squeeze().numpy()  # Remove batch dimension and convert to numpy
        original_image = np.transpose(original_image, (1, 2, 0))  # Convert back to Height x Width x Channels for plotting

        print("Original Image Shape:", original_image.shape)
        # Plotting both original image and saliency map side by side
        plt.figure(figsize=(10, 5))  # Set figure size

        # Plot original image
        n_rows = 1
        n_cols = 2

        plt.subplot(n_rows, n_cols, 1)  # first subplot
        plt.imshow(original_image, cmap='gray') 
        plt.title("Original Image")
        plt.axis('off')

        # Plot saliency map
        plt.subplot(n_rows, n_cols, 2)  # second subplot
        plt.imshow(saliency.cpu(), cmap='hot')
        plt.title(f"Saliency Map (Label: {input_label})")
        plt.axis('off')

        plt.show()
