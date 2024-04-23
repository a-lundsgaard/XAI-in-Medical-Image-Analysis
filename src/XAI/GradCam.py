import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
# from src.XAI.utils.save_plots import save_saliency_maps
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset


class GradCamResnet:
    def __init__(self, modelWrapper: ResNetModel, layerCount=20):
        self.modelWrapper = modelWrapper
        self.fileSaver = PLTSaver(self.__class__.__name__)
        self.heatmap: Tensor = None
        self.layerCount = layerCount

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
    
    def generateMultipleGradCam(self, image_count=1, use_test_data=True, save_output=False, save_dir="default", externalEvalData: TensorDataset = None):
        """
        Generate Grad-CAM visualization for a set of images.
        Args:
            image_count (int): The number of images for which to generate Grad-CAM visualizations.
        """
        self.fileSaver.set_custom_save_dir(save_dir, save_output)

        max_image_count = self.modelWrapper.dataLoader.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count

        for i in range(count):
            self.generate_grad_cam(index=i, use_test_data=use_test_data, externalEvalData=externalEvalData)

    def generate_grad_cam(self, index=0, use_test_data=True, externalEvalData: TensorDataset = None):
        """
        Generate Grad-CAM visualization for a given input image index from the test dataset.
        """
        if externalEvalData is not None:
            input_image, input_label = self.modelWrapper.get_single_image(externalEvalData, index)
        else:
            if use_test_data:
                input_image, input_label = self.modelWrapper.get_single_test_image(index)
            else:
                input_image, input_label = self.modelWrapper.get_single_train_image(index)

        # Find the last convolutional layer
        target_layer = self.__find_last_conv_layer(self.modelWrapper.model)
        if target_layer is None:
            print("No convolutional layer found.")
            return

        # Hook the target layer to access its feature maps and gradients
        features = []
        grads = []

        def hook_features(module, input, output):
            # print("Feature map mean:", output.mean().item())
            features.append(output)

        def hook_grads(module, grad_in, grad_out):
            # print("Gradient mean:", grad_out[0].mean().item())
            grads.append(grad_out[0])

        hook_f = target_layer.register_forward_hook(hook_features)
        hook_b = target_layer.register_backward_hook(hook_grads)

        # Forward pass
        output: Tensor = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        # Backward pass
        if self.modelWrapper.model.fc.out_features == 1:  # Regression
            target: Tensor = output
        else:  # Classification
            target = output.argmax(dim=1)
        target.backward()

        # Remove hooks
        hook_f.remove()
        hook_b.remove()

        # Process the feature maps and gradients
        gradients = grads[0]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        feature_maps = features[0]
        for i in range(feature_maps.shape[1]):
            feature_maps[:, i, :, :] *= pooled_gradients[i]

        # Generate the heatmap
        fe_max = torch.max(feature_maps)
        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        # heatmap = torch.abs(heatmap)  # Use absolute values to consider all activations
       # heatmap = torch.abs(heatmap)
        max_val = torch.max(heatmap)

        # heatmap = F.relu(heatmap)  # already applying ReLU here
        #heatmap = F.relu(heatmap)
        #max_val = torch.max(heatmap)
        #mean_val = torch.mean(heatmap)
        print("Max value in heatmap before ReLU is:", max_val)
        

        if max_val > 0:
            #heatmap = F.relu(heatmap)
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)
        else:
            # heatmap = torch.zeros_like(heatmap)
            #heatmap /= heatmap.norm() + 1e-8
            #heatmap = torch.abs(heatmap)
            #min_val = torch.min(heatmap)
            #heatmap += min_val*0.01

            min_val = torch.min(heatmap).abs()
            print("Min value in heatmap before ReLU is:", min_val)
            heatmap += min_val*0.01
            # heatmap *= -1
            # heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)
            print("Max value in heatmap after ReLU is zero3.")




        # Resize heatmap to the input image size
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(input_image.size(2), input_image.size(3)),
            mode='bilinear',
            align_corners=False
        ).squeeze()


        # Convert heatmap to numpy for visualization
        self.heatmap = heatmap
        heatmap_np = heatmap.cpu().detach().numpy()

        """ plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(features[-1][0, 0].cpu().detach(), cmap='gray')
        plt.title("Sample Feature Map")

        plt.subplot(1, 2, 2)
        feature_gradient = grads[-1][0, 0].cpu().detach()
        plt.imshow(feature_gradient, cmap='gray')
        plt.title("Corresponding Gradient Map")
        plt.show()

        if torch.max(feature_gradient) > 0:
            feature_gradient /= torch.max(feature_gradient)
        plt.imshow(feature_gradient, cmap='gray')
        plt.title("Normalized Gradient Map")
        plt.show() """


        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(input_image.cpu().squeeze(), cmap='gray')
        plt.title("Input Image")
        plt.title(f"Input Image (Label: {input_label})")

        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_np, cmap='jet')
        # plt.title("Grad-CAM")
        plt.title(f"Grad-CAM (Prediction: {round(target.item(), 2)})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        img_np = input_image.cpu().squeeze().numpy()
        plt.imshow(img_np, cmap='gray', interpolation='nearest')
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5, interpolation='nearest')
        plt.title("Overlay")
        plt.axis('off')

        self.fileSaver.handleSaveImage(index, plt, f"grad_cam_{input_label}")

        plt.show()
