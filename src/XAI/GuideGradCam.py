import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.XAI.VanillaSaliency import VanillaSaliency
from src.XAI.GradCam import GradCamResnet
from src.XAI.utils.SaveFiles import PLTSaver

class GuidedGradCam:
    def __init__(self, grad_cam: GradCamResnet, vanilla_saliency: VanillaSaliency):
        self.grad_cam = grad_cam
        self.vanilla_saliency = vanilla_saliency
        self.fileSaver = PLTSaver(self.__class__.__name__)


    def generate_multiple_guided_grad_cam(self, image_count=1, use_test_data=True, save_output=False, save_dir="default"):
        """
        Generate Guided Grad-CAM visualizations for a set of images.
        Args:
            image_count (int): The number of images for which to generate Guided Grad-CAM visualizations.
            use_test_data (bool): Whether to use test data or training data.
            save_output (bool): Whether to save the output visualizations.
            save_dir (str): The directory to save the output visualizations.
        """
        max_image_count = self.grad_cam.modelWrapper.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        self.fileSaver.set_custom_save_dir(save_dir, save_output)

        for i in range(count):
            self.generate_guided_grad_cam(index=i, use_test_data=use_test_data, save_output=save_output)

    def generate_guided_grad_cam(self, index=0, use_test_data=True, save_output=False):
        """
        Generate a Guided Grad-CAM visualization combining Grad-CAM and Vanilla Gradient saliency maps.
        Args:
            index (int): Index of the image in the dataset.
            use_test_data (bool): Whether to use test data or training data.
            save_output (bool): Whether to save the output visualization.
        """
        # Generate Grad-CAM heatmap
        input_image, input_label = self.grad_cam.modelWrapper.get_single_test_image(index) if use_test_data else self.grad_cam.modelWrapper.get_single_train_image(index)
        self.grad_cam.generate_grad_cam(index=index, use_test_data=use_test_data)
        grad_cam_heatmap = self.grad_cam.heatmap  # Assuming heatmap is stored after grad_cam generation

        # Generate Vanilla Gradient saliency map
        self.vanilla_saliency.generate_saliency_map(input_image, input_label, index)
        vanilla_grad = self.vanilla_saliency.saliency  # Assuming saliency is stored after saliency map generation

        # Upsample Grad-CAM heatmap to input image size
        grad_cam_heatmap_resized = F.interpolate(
            grad_cam_heatmap.unsqueeze(0).unsqueeze(0),
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Element-wise multiplication of the upsampled Grad-CAM heatmap with the Vanilla Gradient map
        guided_grad_cam = grad_cam_heatmap_resized * vanilla_grad

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

        if save_output:
            # Save the figure as needed
            pass  # Implement saving logic based on the setup
