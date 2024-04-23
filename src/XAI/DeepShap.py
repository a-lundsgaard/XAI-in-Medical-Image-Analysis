import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.baseModels.resnet_regression import ResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch.utils.data import TensorDataset
import shap

class DeepShapResnet:
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper
        self.fileSaver = PLTSaver(self.__class__.__name__)

    def remove_hooks(self):
        """
        Remove existing backward hooks to avoid conflicts with SHAP.
        """
        handles = []
        for name, module in self.modelWrapper.model.named_modules():
            if hasattr(module, '_backward_hooks'):
                handle = module._backward_hooks.clear()  # Clear existing hooks
                handles.append(handle)
        return handles

    def restore_hooks(self, handles):
        """
        Restore the backward hooks removed previously.
        """
        for handle in handles:
            handle.restore()

    def generate_deep_shap(self, index=0, use_test_data=True, externalEvalData: TensorDataset = None, num_background_samples=50):
        if externalEvalData is not None:
            input_image, input_label = self.modelWrapper.get_single_image(externalEvalData, index)
        else:
            if use_test_data:
                input_image, input_label = self.modelWrapper.get_single_test_image(index)
            else:
                input_image, input_label = self.modelWrapper.get_single_train_image(index)
        input_image = input_image.to(self.modelWrapper.device)
        background = self.modelWrapper.dataLoader.testData.tensors[0][:num_background_samples]
        background = background.to(self.modelWrapper.device)

        # Remove existing hooks if necessary
        handles = self.remove_hooks()

        # Define SHAP Deep Explainer
        explainer = shap.DeepExplainer(self.modelWrapper.model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(input_image.unsqueeze(0))

        # Restore hooks after using SHAP
        # self.restore_hooks(handles)

        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image.cpu().squeeze(), cmap='gray')
        plt.title(f"Input Image (Label: {input_label})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        shap.image_plot(shap_values, input_image.numpy().transpose(1, 2, 0))
        self.fileSaver.handleSaveImage(index, plt, f"deep_shap_{input_label}")
        plt.show()

# Example Usage
# model = ResNetModel(...)
# deep_shap_resnet = DeepShapResnet(model)
# deep_shap_resnet.generate_deep_shap(index=10, use_test_data=True)
