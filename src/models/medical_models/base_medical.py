import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torcheval.metrics import R2Score
from torch import Tensor
import os
from torchvision.models import resnet18  # Adjust as needed
from abc import ABC, abstractmethod
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from scipy.stats import f

class BaseMedical(ABC):
    def __init__(self,
                 num_epochs,
                 data_loader: NiftiDataLoader,
                 learning_rate=0.01,
                 weight_decay=None,
                 dropout_rate=None,
                 depth=18,
                 pretrained=True,
                 model=None,
                 n_input_channels=1,
                 ):

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_loader: NiftiDataLoader = data_loader
        self.model: nn.Module = model
        self.pretrained = pretrained

        self.depth: int = depth
        self.n_input_channels: int = n_input_channels
        self.spatial_dims: int = 2
        self.pretrained_weights_path = f"../src/models/weights/resnet_{self.depth}_23dataset.pth"
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        self.image_shape = None

        # Check what spatial dimensions the first training image is and set the model to that
        print("Data loader train loader: ", len(self.data_loader.train_loader), len(self.data_loader.train_loader.dataset))
        if self.data_loader.train_loader:
            for batch_data in self.data_loader.train_loader:
                # print("Batch data: ", batch_data)
                images = batch_data["image"]
                break
            # print("Image spatial dimensions: ", len(images.shape) - 2)
            print("Image spatial dimensions: ", images.shape)
            self.image_shape = images.shape


            self.spatial_dims = len(images.shape) - 2

        # Set the n_input_channels based on the number of channels in the first image
        if images is not None:
            self.n_input_channels = images.shape[1]
        else:
            print("No training data found. Defaulting to channel.")
        print("Number of input channels: ", self.n_input_channels)

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.set_model()
        self.load_external_weights()

        num_labels = len(self.data_loader.train_loader.dataset[0]['label'])

        num_features = self.model.fc.in_features
        if dropout_rate:
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_labels) 
            )
        else:
            self.model.fc = nn.Linear(num_features, num_labels) 
        self.model.to(self.device)


        # for name, param in self.model.named_parameters():
        #     if 'adaptive_slicing.deviation' in name:
        #         print(f'Found adaptive_slicing.deviation in model parameters: {name} -> {param.data}')

        print("GPU: ", next(self.model.parameters()).device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'))

    def load_external_weights(self):
        pretrained_weights_path = f"../src/models/weights/resnet_{self.depth}_23dataset.pth"
        if self.pretrained and pretrained_weights_path is not None:
            state = torch.load(pretrained_weights_path, map_location=self.device)

            # Handle state dictionary key adjustment
            if "state_dict" in state:
                state = state["state_dict"]

            # Strip the `module.` prefix
            new_state_dict = {}
            for k, v in state.items():
                if k.startswith("module."):
                    new_state_dict[k[len("module."):]] = v  # Remove the `module.` prefix
                else:
                    new_state_dict[k] = v

            print("State dict key adjustment", new_state_dict.keys())

            model_state_dict = self.model.state_dict()
            # Only load weights that match the model's state dictionary keys
            matched_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}
            print("Keys to be loaded into model:", matched_state_dict.keys())

            # Load the matched weights into the model
            model_state_dict.update(matched_state_dict)
            self.model.load_state_dict(model_state_dict)

    @abstractmethod
    def set_model(self):
        raise NotImplementedError
    
    def calculate_p_value(self, r2_score, lenght_of_dataset, n_outputs):
        f_stat = (r2_score * (lenght_of_dataset - n_outputs - 1)) / ((1 - r2_score) * n_outputs)
        p_value = 1 - f.cdf(f_stat, n_outputs, lenght_of_dataset - n_outputs - 1)
        return p_value

    def validation_loss(self):
        self.model.eval()
        total_loss = 0
        count = 0
        r2_metric = R2Score().to(self.device)
        with torch.no_grad():
            for batch_data in self.data_loader.val_loader:
                images = batch_data["image"].to(self.device)
                labels = torch.stack([value.float().to(self.device) for value in batch_data["label"].values()], dim=1)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                count += 1
                r2_metric.update(outputs, labels)
        average_loss = total_loss / count if count > 0 else 0
        r2_score = r2_metric.compute()
        # convert r2_score to float
        r2_score = r2_score.item()

        n = len(self.data_loader.val_loader.dataset)
        k = outputs.shape[1]
        p_value = self.calculate_p_value(r2_score, n, k)

        return average_loss, r2_score, p_value

    def train(self):

        self.data_loader.run_replacement_thread()
        print(f"Number of training images: {len(self.data_loader.train_ds)}")
        data_refresh_count = 1

        best_val_loss = float('inf')
        best_r2_score = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            has_been_saved = False

            for batch_data in self.data_loader.train_loader:
                images = batch_data["image"].to(self.device)
                labels = torch.stack([value.float().to(self.device) for value in batch_data["label"].values()], dim=1)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss: Tensor = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Compute validation loss once per epoch
            current_val_loss, r2_score, p_value = self.validation_loss()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {running_loss / len(self.data_loader.train_loader)}, Val Loss: {current_val_loss}, R^2 Score: {r2_score}, P-value: {p_value}")

            # Log losses to TensorBoard
            self.writer.add_scalar('Loss/train', running_loss / len(self.data_loader.train_loader), epoch)
            self.writer.add_scalar('Loss/val', current_val_loss, epoch)

            if r2_score > best_r2_score:
                best_r2_score = r2_score
                self.save_model(epoch + 1, current_val_loss, r2=best_r2_score, is_best=False)
                has_been_saved = True

            # Save the best model
            if current_val_loss < best_val_loss and not has_been_saved:
                best_val_loss = current_val_loss
                self.save_model(epoch + 1, current_val_loss, r2=best_r2_score, is_best=True)

            # Update cache if needed
            if self.data_loader.replace_rate == 1 and epoch == int(self.num_epochs * self.data_loader.cache_rate * data_refresh_count):
                data_refresh_count += 1
                self.data_loader.update_cache()

        self.data_loader.shutdown_cache()
        self.writer.close()

    def get_single_image(self, data: TensorDataset, index: int = 0):
        if data is not None:
            # Fetch the image and label tensors
            # print("Data: ", data.__getitem__(index)["image"])
            if index < len(data):
                img_tensor = data.__getitem__(index)["image"]
                label_dict: Dict = data.__getitem__(index)["label"]
                # Add a batch dimension, convert to float, and move to the correct device
                print(label_dict)
                img_tensor = img_tensor.cpu()

                label_value = [value for value in label_dict.values()][0]
                # Move the label to the correct device
                return img_tensor, label_value
            
    def get_single_test_image(self, index=0):
        return self.get_single_image(self.data_loader.test_ds, index)
    
    def get_single_train_image(self, index=0):
        return self.get_single_image(self.data_loader.train_ds, index)

    def evaluate(self, loader=None):
        self.model.eval()
        total_loss = 0.0
        r2_metric = R2Score().to(self.device)
        data_loader = loader if loader else self.data_loader.test_loader
        with torch.no_grad():
            for batch_data in data_loader:
                images = batch_data["image"].to(self.device)
                labels = torch.stack([value.float().to(self.device) for value in batch_data["label"].values()], dim=1)
                predicted = self.model(images)

                loss = self.criterion(predicted, labels)
                total_loss += loss.item()
                r2_metric.update(predicted, labels)

        r2_score = r2_metric.compute()
        r2_score = r2_score.item()

        n = len(data_loader.dataset)
        k = predicted.shape[1]

        p_value = self.calculate_p_value(r2_score, n, k)
        
        print(f'R^2 score of the network on the test images: {r2_score}, p-value: {p_value}')
        print(f"Test Loss: {total_loss / len(self.data_loader.test_loader)}")

        # for name, param in self.model.named_parameters():
        #     if 'adaptive_slicing.deviation' in name:
        #         print(f'Found adaptive_slicing.deviation in model parameters: {name} -> {param.data}')

        # Log evaluation metrics to TensorBoard
        self.writer.add_scalar('Loss/test', total_loss / len(self.data_loader.test_loader), 0)
        self.writer.add_scalar('R2Score/test', r2_score, 0)
        self.writer.close()

    def save_model(self, epoch, val_loss, r2, is_best=False):

        os.makedirs(self.save_dir, exist_ok=True)
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'r2_score': r2,
        }
        shape = self.image_shape
        save_path = os.path.join(self.save_dir, f"{self.__class__.__name__}_{self.depth}_{len(self.data_loader.train_ds)}_height_{shape}_epoch_{epoch}_val_{round(val_loss, 2)}_r2_{round(r2, 2)}.pth")
        torch.save(model_state, save_path)
        print(f"Model saved at {save_path}")

    def load_model(self, model_name: str):

        path = os.path.join(self.save_dir, model_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at '{path}'")

        checkpoint: dict = torch.load(path, map_location=self.device)
        #self.model.load_state_dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        print(f"Model loaded from {path}, epoch: {epoch}, val_loss: {val_loss}")
