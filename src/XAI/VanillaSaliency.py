import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
# Assuming ResNetModel is correctly imported from your project structure
from src.models.baseModels.resnet_regression import ResNetModel
from torch import Tensor
from src.XAI.utils.SaveFiles import PLTSaver
from src.XAI.utils.BaseXAI import BaseXAI
from torch.utils.data import TensorDataset

class VanillaSaliency(BaseXAI):
    def __init__(self, modelWrapper: ResNetModel):
        super().__init__(modelWrapper)

    def get_saliency_maps(self, image_count=1, save_output=False, save_dir="default"):
        """
        Generate saliency maps for a set of images.
        """
        self.fileSaver.set_custom_save_dir(save_dir, save_output)
        max_image_count = self.modelWrapper.dataLoader.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count
        for i in range(count):
            input_image, input_label = self.modelWrapper.get_single_test_image(index=i)
            self.generate_map(input_image, input_label, i=i)
    
    # def generate_map(self, input_image: Tensor, input_label, i=0):
    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir=None, externalEvalData: TensorDataset = None, plot=True):
        
        input_image, input_label = self.get_image_and_label(index, use_test_data, externalEvalData)

        """
        Generate a saliency map for a given input image and label.
        """
        input_image = input_image.to(self.device).requires_grad_(True)
        output = self.modelWrapper.model(input_image)  
        self.modelWrapper.model.zero_grad() 
        target: Tensor = output[0]
        target.backward()
        saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)
        saliency = saliency.reshape(self.modelWrapper.dataLoader.image_size)
        self.heatmap = saliency
        original_image = input_image.detach().cpu().squeeze().numpy()

        if plot:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title(f"Input Image (Label: {input_label})")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(saliency.cpu(), cmap='hot')
            plt.title(f"Saliency Map (Prediction: {round(target.item(), 2)})")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(original_image, cmap='gray', interpolation='nearest')
            plt.imshow(saliency.cpu(), cmap='hot', alpha=0.5, interpolation='nearest')  # Overlay with transparency
            plt.title("Overlay of Saliency Map")
            plt.axis('off')
            
            if save_dir and save_output:
                self.fileSaver.set_custom_save_dir(save_dir, save_output)
                self.fileSaver.handleSaveImage(id=index, plt=plt, name=input_label)

            print(f"Image {index+1}")
            plt.show()
