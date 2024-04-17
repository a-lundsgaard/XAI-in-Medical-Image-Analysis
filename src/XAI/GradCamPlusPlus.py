import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.models.baseModels.resnet_regression import ResNetModel
# from src.XAI.utils.save_plots import save_saliency_maps
from src.XAI.utils.SaveFiles import PLTSaver
from torch import Tensor
from torch.utils.data import TensorDataset


class GradCamPlusPlus:
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
    
    def normalize_map(self, grad_cam_map):
        min_val = grad_cam_map.min()
        max_val = grad_cam_map.max()
        if max_val - min_val > 0:  # Avoid division by zero
            normalized_map = (grad_cam_map - min_val) / (max_val - min_val)
        else:
            normalized_map = torch.zeros_like(grad_cam_map)  # If all values are the same
        return normalized_map

    def generateMultipleGradCam(self, image_count=1, use_test_data=True, save_output=False, save_dir="default", externalEvalData: TensorDataset = None):
        self.fileSaver.set_custom_save_dir(save_dir, save_output)
        max_image_count = self.modelWrapper.dataLoader.testData.tensors[0].shape[0]
        count = image_count if image_count <= max_image_count else max_image_count
        for i in range(count):
            self.generate_grad_cam(index=i, use_test_data=use_test_data, externalEvalData=externalEvalData)

    def generate_grad_cam(self, index=0, use_test_data=True, externalEvalData: TensorDataset = None):
        if externalEvalData is not None:
            input_image, input_label = self.modelWrapper.get_single_image(externalEvalData, index)
        else:
            if use_test_data:
                input_image, input_label = self.modelWrapper.get_single_test_image(index)
            else:
                input_image, input_label = self.modelWrapper.get_single_train_image(index)

        target_layer = self.__find_last_conv_layer(self.modelWrapper.model)
        if target_layer is None:
            print("No convolutional layer found.")
            return

        features = []
        grads = []

        def hook_features(module, input, output):
            features.append(output)

        def hook_grads(module, grad_in, grad_out):
            grads.append(grad_out[0])

        hook_f = target_layer.register_forward_hook(hook_features)
        hook_b = target_layer.register_backward_hook(hook_grads)

        output: Tensor = self.modelWrapper.model(input_image)
        self.modelWrapper.model.zero_grad()

        if self.modelWrapper.model.fc.out_features == 1:
            target: Tensor = output
        else:
            target = output.argmax(dim=1)
        target.backward()

        target: Tensor = output

        hook_f.remove()
        hook_b.remove()

        gradients = grads[0]
        mean_grads = torch.mean(gradients, dim=[0, 1, 2, 3])
        feature_maps = features[0]
        fe_max = feature_maps.max()
        weights = F.adaptive_avg_pool2d(gradients, (1, 1))

        grad_cam_map = torch.zeros(feature_maps.shape[2:], device=feature_maps.device)
        for i in range(feature_maps.shape[1]):
            grad_cam_map += weights[0, i, :, :] * feature_maps[0, i, :, :]

        #grad_cam_map = self.normalize_map(grad_cam_map)
        max_val = grad_cam_map.max()

        if max_val > 0:
            grad_cam_map = F.relu(grad_cam_map)
            grad_cam_map = grad_cam_map / grad_cam_map.max()
        else:
            grad_cam_map = torch.abs(grad_cam_map)
            max_val = torch.max(grad_cam_map)
            grad_cam_map -= max_val*0.95
            #print("Min value in heatmap before ReLU is:", min_val)
            #grad_cam_map += min_val
            # heatmap *= -  1
            grad_cam_map = F.relu(grad_cam_map)
            grad_cam_map /= torch.max(grad_cam_map)
            print("Max value in heatmap after ReLU is zero3.")            



        heatmap = F.interpolate(
            grad_cam_map.unsqueeze(0).unsqueeze(0),
            size=(input_image.size(2), input_image.size(3)),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        self.heatmap = heatmap
        heatmap_np = heatmap.cpu().detach().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(input_image.cpu().squeeze(), cmap='gray')
        plt.title(f"Input Image (Label: {input_label})")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_np, cmap='jet')
        plt.title(f"Grad-CAM++ (Prediction: {round(target.item(), 2)})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        img_np = input_image.cpu().squeeze().numpy()
        plt.imshow(img_np, cmap='gray', interpolation='nearest')
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5, interpolation='nearest')
        plt.title("Overlay")
        plt.axis('off')

        self.fileSaver.handleSaveImage(index, plt, f"grad_cam_{input_label}")

        plt.show()