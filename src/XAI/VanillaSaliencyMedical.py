import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.medical_models.combined_medical import MedicalCombinedResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset
from src.XAI.utils.BaseXAI import BaseXAI

class VanillaSaliency(BaseXAI):
    def __init__(self, modelWrapper: MedicalCombinedResNetModel, views=['Coronal', 'Axial', 'Sagittal']):
        super().__init__(modelWrapper)
        self.views = views
        self.modelWrapper.model.eval()

    
    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir=None, externalEvalData: TensorDataset = None, plot=True):
        input_image, input_label = self.get_image_and_label(index, use_test_data, externalEvalData)
        input_image = input_image.unsqueeze(0).to(self.device).requires_grad_(True)

        # Forward pass
        output = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        # Backward pass
        target = output[0]
        target.backward()


        # Generate saliency map for each view
        saliencies = input_image.grad.abs().cpu().detach().numpy().squeeze()
        print("saliencies.shape", saliencies.shape)

        n_input_channels = saliencies.shape[0]
        view_titles = self.views
        channels_per_view = n_input_channels // len(view_titles)
        print("channels_per_view", channels_per_view)


        original_images = input_image.cpu().detach().numpy().squeeze()

        # Visualization
        if plot:
            fig, axes = plt.subplots(len(view_titles), channels_per_view+1, figsize=(24, 18))
            # print("axes.shape", axes.shape)

            if len(view_titles) == 1:
                axes = axes[np.newaxis, :]

            for i in range(len(view_titles)):
                print("i", i)
                for j in range(channels_per_view):
                    channel_idx = i * channels_per_view + j
                    original_image = original_images[channel_idx]
                    saliency = saliencies[channel_idx]

                    axes[i, j].imshow(original_image, cmap='gray')
                    axes[i, j].imshow(saliency, cmap='hot', alpha=0.5)
                    axes[i, j].set_title(f"{view_titles[i]} - Channel {j + 1}")
                    axes[i, j].axis('off')

                # Combined overlay for the current view
                combined_image = np.mean([original_images[i * channels_per_view + j] for j in range(channels_per_view)], axis=0)
                combined_saliency = np.mean([saliencies[i * channels_per_view + j] for j in range(channels_per_view)], axis=0)

                axes[i, channels_per_view].imshow(combined_image, cmap='gray')
                axes[i, channels_per_view].imshow(combined_saliency, cmap='hot', alpha=0.5)
                axes[i, channels_per_view].set_title(f"{view_titles[i]} - Combined. Label: {input_label} Prediction: {round(target.item(), 2)}")
                axes[i, channels_per_view].axis('off')

            if save_dir and save_output:
                self.fileSaver.set_custom_save_dir(save_dir, save_output)
                self.fileSaver.handleSaveImage(index, plt, f"saliency_map_{input_label}")

            plt.tight_layout()
            plt.show()