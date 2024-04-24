import os
from abc import ABC, abstractmethod
import cv2
from torch import Tensor
from src.XAI.utils.SaveFiles import PLTSaver
from src.models.baseModels.resnet_regression import ResNetModel
from torch.utils.data import TensorDataset
import torch


class BaseXAI(ABC):
    """
    Abstract class for generating images.
    """
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper
        self.fileSaver = PLTSaver(self.__class__.__name__)
        self.heatmap: Tensor = None
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def find_last_conv_layer(self, model: torch.nn.Module):
        """
        Find the last convolutional layer in the model for Grad-CAM.
        """
        conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module

        if conv_layer is None:
            raise ValueError("No convolutional layer found.")
        return conv_layer

    @abstractmethod
    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir="default", externalEvalData: TensorDataset = None, plot=True):
        """
        Generates a set of images. This method needs to be implemented by subclasses.
        """
        pass

    def get_image_and_label(self, index=0, use_test_data=True, externalEvalData: TensorDataset = None):
        if externalEvalData is not None:
            input_image, input_label = self.modelWrapper.get_single_image(externalEvalData, index)
        else:
            if use_test_data:
                input_image, input_label = self.modelWrapper.get_single_test_image(index)
            else:
                input_image, input_label = self.modelWrapper.get_single_train_image(index)
        return input_image, input_label
