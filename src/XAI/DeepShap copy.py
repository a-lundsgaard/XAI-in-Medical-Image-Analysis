import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.baseModels.resnet_regression import ResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch.utils.data import TensorDataset
import shap
# reload shap in case it was modified
import importlib

importlib.reload(shap)

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


    def __find_last_conv_layer(self, model: torch.nn.Module):
        """
        Find the last convolutional layer in the model for Grad-CAM.
        """
        conv_layer = None
        instanceCount = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                instanceCount += 1
                conv_layer = module
                if instanceCount > self.layerCount-1:
                    break
        print(f"Found {instanceCount} instances of Conv2d layers.") 
        return conv_layer

    def generate_deep_shap(self, index=0, use_test_data=True, externalEvalData: TensorDataset = None, num_background_samples=10):
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
        #hooks = self.remove_hooks()

        # Define SHAP Deep Explainer and generate SHAP values
        # explainer = shap.GradientExplainer(self.modelWrapper.model, background)
        explainer = shap.GradientExplainer((self.modelWrapper.model, self.modelWrapper.model.eval()), background)

        shap_values = explainer.shap_values(input_image , nsamples=10)
        print(shap_values.shape)


        #shap_values = explainer.shap_values(input_image)

        # Restore hooks if they were removed before
        #self.restore_hooks(hooks)




        # Visualization

        # Assuming input_image originally has shape [1, 1, 256, 256] due to unsqueeze and single channel
        # Removing batch and channel dimensions since it's single-channel grayscale
        image_for_plot = input_image.cpu().squeeze().numpy() # This will result in shape [256, 256]
        print(image_for_plot.shape)
        shap.image_plot(shap_values, image_for_plot)
        # self.fileSaver.handleSaveImage(index, plt, f"deep_shap_{input_label}")
        # plt.show()

# Example Usage
# model = ResNetModel(...)
# deep_shap_resnet = DeepShapResnet(model)
# deep_shap_resnet.generate_deep_shap(index=10, use_test_data=True)
