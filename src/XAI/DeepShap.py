import shap.maskers
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch.utils.data import TensorDataset
import shap
from shap import GradientExplainer

class DeepShapResnet:
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper
        self.fileSaver = PLTSaver(self.__class__.__name__)

    def __find_last_conv_layer(self, model: torch.nn.Module):
        """
        Find the last convolutional layer in the model for Grad-CAM.
        """
        conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module
        print(f"Found {conv_layer} instances of Conv2d layers.") 
        return conv_layer

    def generate_shap_values(self, image_count=1, use_test_data=True, save_output=False, save_dir="default", externalEvalData: TensorDataset = None):
        """
        Generate SHAP values for a set of images.
        """
        self.fileSaver.set_custom_save_dir(save_dir, save_output)

        max_image_count = self.modelWrapper.dataLoader.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        # Initialize SHAP GradientExplainer
        background, _ = next(iter(self.modelWrapper.dataLoader.train_loader))
        background = background.to(self.modelWrapper.device)[:1000]  # Use up to 100 examples for background
        explainer = shap.GradientExplainer(self.modelWrapper.model, background)

        for i in range(count):
            self.generate_shap_image(index=i, explainer=explainer, use_test_data=use_test_data, externalEvalData=externalEvalData)

    def generate_shap_image(self, index=0, explainer: GradientExplainer= None, use_test_data=True, externalEvalData: TensorDataset = None):
        """
        Generate SHAP visualization for a given input image index from the dataset.
        """
        if externalEvalData is not None:
            input_image, input_label = self.modelWrapper.get_single_image(externalEvalData, index)
        else:
            if use_test_data:
                input_image, input_label = self.modelWrapper.get_single_test_image(index)
            else:
                input_image, input_label = self.modelWrapper.get_single_train_image(index)

        # Generate SHAP values
        shap_values = explainer.shap_values(input_image)
        shap.image_plot(shap_values[0], -input_image.cpu().numpy())
        

        # Visualization
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 3, 1)
        # plt.imshow(input_image.cpu().squeeze(), cmap='gray')
        # plt.title(f"Input Image (Label: {input_label})")
        # plt.axis('off')

        # plt.subplot(1, 3, 2)

        # plt.subplot(1, 3, 3)
        # # Overlay original image and SHAP heatmap
        # img_np = input_image.cpu().squeeze().numpy()
        # shap_np = shap_values[0].sum(0)
        # plt.imshow(img_np, cmap='gray', interpolation='nearest')
        # plt.imshow(shap_np, cmap='jet', alpha=0.5, interpolation='nearest')
        # plt.title("Overlay")
        # plt.axis('off')

        # self.fileSaver.handleSaveImage(index, plt, f"shap_{input_label}")

        # plt.show()
