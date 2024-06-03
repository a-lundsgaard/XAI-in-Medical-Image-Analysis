import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import ipyvolume as ipv
from src.models.medical_models.combined_medical import MedicalCombinedResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset
from src.XAI.utils.BaseXAI import BaseXAI


class VanillaSaliency3D(BaseXAI):
    def __init__(self, modelWrapper: MedicalCombinedResNetModel):
        super().__init__(modelWrapper)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir=None, externalEvalData: TensorDataset = None, plot=True):
        input_image, input_label = self.get_image_and_label(
            index, use_test_data, externalEvalData)
        input_image = input_image.unsqueeze(0).to(
            self.device).requires_grad_(True)

        # Forward pass
        output = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        # Backward pass
        target = output[0]
        target.backward()

        # Generate saliency map for the entire 3D image
        saliencies = input_image.grad.abs().cpu().detach().numpy().squeeze()
        print("saliencies.shape", saliencies.shape)

        original_images = input_image.cpu().detach().numpy().squeeze()

        # Normalize original images and saliency map for better visualization
        original_images = (original_images - np.min(original_images)) / \
            (np.max(original_images) - np.min(original_images))
        saliencies = (saliencies - np.min(saliencies)) / \
            (np.max(saliencies) - np.min(saliencies))

        # 3D Visualization with ipyvolume
        if plot:
            # 2D Visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # Take the middle slice for visualization
            # slice_idx = original_images.shape[1] // 2
            print("original_images.shape", original_images.shape)

            slice_x = original_images.shape[0] // 2
            slice_y = original_images.shape[1] // 2
            slice_z = original_images.shape[2] // 2

            for i, axis in enumerate(['x', 'y', 'z']):
                if axis == 'x':
                    img_slice = original_images[slice_x, :, :]
                    saliency_slice = saliencies[slice_x, :, :]
                elif axis == 'y':
                    img_slice = original_images[:, slice_y, :]
                    saliency_slice = saliencies[:, slice_y, :]
                else:
                    axis == 'z'
                    img_slice = original_images[:, :, slice_z]
                    saliency_slice = saliencies[:, :, slice_z]

                axes[i].imshow(img_slice, cmap='gray')
                axes[i].imshow(saliency_slice, cmap='hot', alpha=0.5)
                axes[i].set_title(f"{axis.upper()}-axis slice")
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

                    # Create RGBA volumes
            # rgba_original = np.zeros(original_images.shape + (4,))
            # rgba_saliency = np.zeros(saliencies.shape + (4,))

            # # Red channel for original image
            # rgba_original[..., 0] = original_images
            # rgba_original[..., 3] = 0.3  # Set alpha for original image

            # rgba_saliency[..., 1] = saliencies  # Green channel for saliency
            # rgba_saliency[..., 3] = 0.6  # Set alpha for saliency

            # # Combine both volumes
            # combined_volume = np.maximum(rgba_original, rgba_saliency)

            # # Separate intensity and alpha channels
            # intensity = combined_volume[..., :3].max(
            #     axis=-1)  # Max intensity from RGBA channels
            # alpha = combined_volume[..., 3]  # Alpha channel

            # ipv.figure()
            # ipv.volshow(intensity,
            #             level=[0.1, 0.5, 0.8],
            #             opacity=[0.1, 0.5, 0.8],

            #             # level=[0.5, 0.5, 0.5],
            #             # opacity=[0.3, 0.3, 0.3],
            #             # level=0.5,
            #             # opacity=0.3,
            #             controls=True,
            #             extent=[[0, intensity.shape[2]], [0, intensity.shape[1]], [0, intensity.shape[0]]])
            # ipv.show()

        if save_dir and save_output:
            self.fileSaver.set_custom_save_dir(save_dir, save_output)
            self.fileSaver.handleSaveImage(
                index, plt, f"saliency_map_{input_label}")

    def generateMultipleSaliencyMaps(self, image_count=1, use_test_data=True, save_output=False, save_dir="default", externalEvalData: TensorDataset = None):
        """
        Generate saliency map visualizations for a set of images.
        Args:
            image_count (int): The number of images for which to generate saliency maps.
        """
        self.fileSaver.set_custom_save_dir(save_dir, save_output)

        max_image_count = self.modelWrapper.dataLoader.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        for i in range(count):
            self.generate_map(index=i, use_test_data=use_test_data,
                              externalEvalData=externalEvalData)
