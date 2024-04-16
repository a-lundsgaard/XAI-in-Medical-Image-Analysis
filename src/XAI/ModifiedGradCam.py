import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
# from src.XAI.utils.save_plots import save_saliency_maps
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset

class ModifiedGradCam:
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper
        self.fileSaver = PLTSaver(self.__class__.__name__)
        self.heatmap: Tensor = None
    
    def __find_last_conv_layer(self, model: torch.nn.Module):
        conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module
        return conv_layer

    def generate_grad_cam(self, index=0, use_test_data=True):
        input_image, true_value = self.modelWrapper.get_single_test_image(index) if use_test_data else self.modelWrapper.get_single_train_image(index)
        target_layer = self.__find_last_conv_layer(self.modelWrapper.model)
        if not target_layer:
            print("No convolutional layer found.")
            return

        features, grads = [], []
        hook_f = target_layer.register_forward_hook(lambda m, i, o: features.append(o))
        hook_b = target_layer.register_backward_hook(lambda m, gi, go: grads.append(go[0]))

        output: Tensor = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        predicted_value = output  # assuming the output is a single value for regression
        loss: Tensor = 1 / (predicted_value - true_value).pow(2)  # Use inverse squared error
        loss.backward()

        hook_f.remove()
        hook_b.remove()

        gradients = grads[0]
        feature_maps = features[0]
        weights = F.adaptive_avg_pool2d(gradients, (1, 1))

        grad_cam_map = (feature_maps * weights).sum(dim=1).clamp(min=0)  # Using ReLU to clamp negative values
        grad_cam_map /= grad_cam_map.max()  # Normalize

        heatmap = F.interpolate(grad_cam_map.unsqueeze(0).unsqueeze(0), size=input_image.size()[2:], mode='bilinear', align_corners=False).squeeze()
        self.heatmap = heatmap

        # Visualization code as in the original function...

        self.fileSaver.handleSaveImage(index, plt, f"modified_grad_cam_{true_value.item()}")

        plt.show()
