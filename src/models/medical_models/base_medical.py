import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torcheval.metrics import R2Score
from torch import Tensor
import os
from torchvision.models import ResNet
from abc import ABC, abstractmethod
from typing import Any
from torch.utils.tensorboard import SummaryWriter

class MedicalResNetModelBase(ABC):
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
        self.spacial_dims: int = 2
        self.pretrained_weights_path = f"../src/models/weights/resnet_{self.depth}_23dataset.pth"
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

        # check what spatial dimensions the first training image is and set the model to that
        if self.data_loader.train_loader:
            for batch_data in self.data_loader.train_loader:
                images = batch_data["image"]
                break
            print("Image spatial dimensions: ", len(images.shape) - 2)
            self.spacial_dims = len(images.shape) - 2

        # set the n_input_channels based on the number of channels in the first image
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

        num_features = self.model.fc.in_features
        if dropout_rate:
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 1)
            )
        else:
            self.model.fc = nn.Linear(num_features, 1)  # Output layer for regression.
        self.model.to(self.device)

        print("gpu: ", next(self.model.parameters()).device)
        
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
            self.model.load_state_dict(new_state_dict)

    @abstractmethod
    def set_model(self):
        raise NotImplementedError

    def validation_loss(self):
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch_data in self.data_loader.val_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].float().to(self.device)
                outputs = self.model(images).flatten()
                loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()
                count += 1
        average_loss = total_loss / count if count > 0 else 0
        return average_loss

    def train(self):

        self.data_loader.run_replacement_thread()
        print("Is cuda available: ", torch.cuda.is_available(), self.device)
        # print no of training images
        print(f"Number of training images: {len(self.data_loader.train_ds)}")
        data_refresh_count = 1

        best_val_loss = float('inf')        

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for batch_data in self.data_loader.train_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].float().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images).flatten()
                loss: Tensor = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Compute validation loss once per epoch
            current_val_loss = self.validation_loss()

            # Print epoch statistics
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {running_loss / len(self.data_loader.train_loader)}, Val Loss: {current_val_loss}")

            # Log losses to TensorBoard
            self.writer.add_scalar('Loss/train', running_loss / len(self.data_loader.train_loader), epoch)
            self.writer.add_scalar('Loss/val', current_val_loss, epoch)

            # Save the best model
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.save_model(epoch + 1, current_val_loss, is_best=True)

            # Update cache if needed
            if self.data_loader.replace_rate == 1 and epoch == int(self.num_epochs * self.data_loader.cache_rate * data_refresh_count):
                data_refresh_count += 1
                self.data_loader.update_cache()

        self.data_loader.shutdown_cache()
        self.writer.close()

    def evaluate(self, loader = None):
        self.model.eval()
        total_loss = 0.0
        r2_metric = R2Score().to(self.device)
        with torch.no_grad():
            for batch_data in loader if loader else self.data_loader.test_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].float().to(self.device)
                predicted = self.model(images).flatten()

                loss = self.criterion(predicted, labels)
                total_loss += loss.item()
                r2_metric.update(predicted, labels)
                # print(f"Predicted: {predicted}, Actual: {labels}")

        r2_score = r2_metric.compute()
        print(f'R^2 score of the network on the test images: {r2_score}')
        print(f"Test Loss: {total_loss / len(self.data_loader.test_loader)}")

        # Log evaluation metrics to TensorBoard
        self.writer.add_scalar('Loss/test', total_loss / len(self.data_loader.test_loader), 0)
        self.writer.add_scalar('R2Score/test', r2_score, 0)
        self.writer.close()

    def save_model(self, epoch, val_loss, is_best=False):

        os.makedirs(self.save_dir, exist_ok=True)
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        height, width = self.data_loader.train_ds[0]["image"].shape[1:]
        save_path = os.path.join(self.save_dir, f"{self.__class__.__name__}_{self.depth}_{len(self.data_loader.train_ds)}_height_{height}_epoch_{epoch}_val_{round(val_loss, 2)}.pth")
        torch.save(model_state, save_path)
        print(f"Model saved at {save_path}")

    def load_model(self, model_name: str):

        path = os.path.join(self.save_dir, model_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at '{path}'")

        checkpoint: dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))

        print(f"Model and optimizer state loaded from '{path}'")
        return epoch, val_loss
