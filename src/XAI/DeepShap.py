import shap.maskers
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
from src.XAI.utils.SaveFiles import PLTSaver
from torch.utils.data import TensorDataset
import shap
from src.XAI.utils.BaseXAI import BaseXAI

class DeepShapResnet(BaseXAI):
    def __init__(self, modelWrapper: ResNetModel):
        self.modelWrapper = modelWrapper

    def generate_shap_values(self):
        """
        Generate SHAP values for a set of images.
        """
        # Initialize SHAP GradientExplainer
        background, _ = next(iter(self.modelWrapper.dataLoader.train_loader))
        background = background.to(self.modelWrapper.device)[:1000]  # Use up to 100 examples for background
        explainer = shap.GradientExplainer(self.modelWrapper.model, background)

        return explainer


    def generate_map(self, index=0, use_test_data=True, save_output=False, save_dir=None, externalEvalData: TensorDataset = None, plot=True):
        """
        Generate SHAP visualization for a given input image index from the dataset.
        """
        input_image, input_label = self.get_image_and_label(index, use_test_data, externalEvalData)

        explainer = self.generate_shap_values()
        shap_values = explainer.shap_values(input_image)
        self.heatmap = shap_values
        
        if plot:
            to_explain = input_image.cpu().numpy()
            # shap.image_plot(shap_values[0], -to_explain, ["Original Image", "SHAP Values"])
                    # Ensure image is in the correct format (height, width, channels)
            # Display SHAP values overlaying the original image
            shap.image_plot(shap_values[0], -to_explain, [f"SHAP Values for Label {input_label}"])
        

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
