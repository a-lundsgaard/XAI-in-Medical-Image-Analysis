import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from models.baseModels.resnet_regression import ResNetModel

class XAIResNet:
    def __init__(self, modelWrapper: ResNetModel, device):
        self.modelWrapper = modelWrapper
        self.device = device
    
    def generate_saliency_map(self, input_image, input_label):
        """
        Generate a saliency map for a given input image and label.
        Args:
            input_image (Tensor): The input image tensor.
            input_label (int): The label of the input image.
        """
        # self.model.eval()
        # Ensure image is on the correct device and set requires_grad
        input_image = input_image.to(self.device).requires_grad_(True)
        
        # Forward pass
        # output = self.model(input_image.unsqueeze(0))  # Add batch dimension
        output = self.modelWrapper.model(input_image)  # input_image is already batched

        self.modelWrapper.model.zero_grad()
        
        # Target for backprop
        target = output[0][input_label]
        
        # Backward pass
        target.backward()
        
        # Saliency map
        saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)  # Take the max across the channels
        saliency = saliency.reshape(self.modelWrapper.image_size)  # Assuming model has attribute image_size
        
        # Plot
        plt.imshow(saliency.cpu(), cmap='hot')
        plt.title(f'Saliency Map for label: {input_label}')
        plt.axis('off')
        plt.show()

    @staticmethod
    def preprocess_image(img_path, image_size):
        """
        Preprocess the image for model input.
        Args:
            img_path (str): Path to the image.
            image_size (tuple): The size to which the image is resized.
        Returns:
            Tensor: The preprocessed image tensor.
        """
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size)
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))  # Convert to channel-first
        img = img / 255.0  # Normalize
        return torch.tensor(img, dtype=torch.float)

# Usage example
# xai_resnet = XAIResNet(model=model.model, device=model.device)  # Assuming 'model' is an instance of ResNetModel class
# input_image = XAIResNet.preprocess_image('<path_to_your_image>', model.image_size)
# input_label = 0  # Assuming you know the label of the image or what you're trying to analyze

# xai_resnet.generate_saliency_map(input_image, input_label)
