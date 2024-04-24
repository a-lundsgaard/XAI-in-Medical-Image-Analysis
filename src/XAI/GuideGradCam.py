import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.XAI.VanillaSaliency import VanillaSaliency
from src.XAI.GradCam import GradCamResnet
from src.XAI.utils.SaveFiles import PLTSaver
from src.XAI.utils.BaseXAI import BaseXAI
from src.models.baseModels.resnet_regression import ResNetModel
from torch.utils.data import TensorDataset

class GuidedGradCam(BaseXAI):
    def __init__(self, grad_cam: GradCamResnet, vanilla_saliency: VanillaSaliency):
        self.grad_cam = grad_cam
        self.vanilla_saliency = vanilla_saliency
        self.modelWrapper = grad_cam.modelWrapper


    def generate_multiple_guided_grad_cam(self, image_count=1, use_test_data=True, save_output=False, save_dir="default"):
        """
        Generate Guided Grad-CAM visualizations for a set of images.
        Args:
            image_count (int): The number of images for which to generate Guided Grad-CAM visualizations.
            use_test_data (bool): Whether to use test data or training data.
            save_output (bool): Whether to save the output visualizations.
            save_dir (str): The directory to save the output visualizations.
        """
        max_image_count = self.grad_cam.modelWrapper.dataLoader.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        self.fileSaver.set_custom_save_dir(save_dir, save_output)

        for i in range(count):
            self.generate_map(index=i, use_test_data=use_test_data, save_output=save_output)

    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir=None, externalEvalData: TensorDataset = None, plot=True):
        """
        Generate a Guided Grad-CAM visualization combining Grad-CAM and Vanilla Gradient saliency maps.
        Args:
            index (int): Index of the image in the dataset.
            use_test_data (bool): Whether to use test data or training data.
            save_output (bool): Whether to save the output visualization.
        """
        # Generate Grad-CAM heatmap
        input_image, input_label = self.get_image_and_label(index, use_test_data, externalEvalData)
        
        self.grad_cam.generate_map(index=index, plot=False)
        self.vanilla_saliency.generate_map(index=index, plot=False)

        grad_cam_heatmap = self.grad_cam.heatmap  # Assuming heatmap is stored after grad_cam generation
        vanilla_grad = self.vanilla_saliency.heatmap  # Assuming saliency is stored after saliency map generation

        # Upsample Grad-CAM heatmap to input image size
        grad_cam_heatmap_resized = F.interpolate(
            grad_cam_heatmap.unsqueeze(0).unsqueeze(0),
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Element-wise multiplication of the upsampled Grad-CAM heatmap with the Vanilla Gradient map
        guided_grad_cam = grad_cam_heatmap_resized * vanilla_grad

        if plot:
            # Visualization
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(input_image.cpu().squeeze().detach().numpy(), cmap='gray')
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(vanilla_grad.cpu().detach().numpy(), cmap='hot')
            plt.title("Vanilla Gradient Saliency")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(guided_grad_cam.cpu().detach().numpy(), cmap='jet')
            plt.title("Guided Grad-CAM")
            plt.axis('off')

            plt.show()

            if save_output and save_dir:
                self.fileSaver.set_custom_save_dir(save_dir, save_output)
                self.fileSaver.handleSaveImage(index, plt, f"guided_grad_cam_{input_label}")
