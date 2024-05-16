# from monai.networks.nets import ResNet, resnet18, resnet34, resnet50
import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torcheval.metrics import R2Score
from torch import Tensor
import os

from torchvision.models import ResNet
from abc import ABC, abstractmethod
# import Any
from typing import Any


class MedicalResNetModelBase(ABC):
    def __init__(self,
                 num_epochs,
                 data_loader: NiftiDataLoader,
                 learning_rate=0.01,
                 weight_decay=None,
                 dropout_rate=None,
                 depth=18,
                 pretrained=True,
                 model = None
                 ):

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_loader: NiftiDataLoader = data_loader
        self.model: nn.Module = model
        self.pretrained = pretrained

        self.depth: int = depth
        self.n_input_channels: int = 1
        self.spacial_dims: int = 2
        self.pretrained_weights_path = f"../src/models/weights/resnet_{self.depth}_23dataset.pth"

        # check what spatial dimensions the first training image is and set the model to that
        if self.data_loader.train_loader:
            for batch_data in self.data_loader.train_loader:
                images = batch_data["image"]
                break
            print("Image spatial dimensions: ", len(images.shape) - 2)
            self.spacial_dims = len(images.shape) - 2

        # set the n_input_channels based on the number of channels in the first image
        self.n_input_channels = images.shape[1]
        print("Number of input channels: ", self.n_input_channels)

                # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # self.device = torch.device("cpu")

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

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            # print("Starting new epoch...")
            self.model.train()
            running_loss = 0.0
            running_val_loss = 0.0

            # print batch size and number of batches
            # print("Number of batches: ", len(self.data_loader.train_loader)) 

            for batch_data in self.data_loader.train_loader:

                # print("Moving images and labels to device")
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].float().to(self.device)


                self.optimizer.zero_grad()
                # print("Performing forward pass...")
                outputs = self.model(images).flatten() # Flatten the output to match the shape of the labels
                # print("Forward pass outputs: ", outputs)
                loss: Tensor = self.criterion(outputs, labels)
                # print("Performing backward pass...")
                loss.backward()
                # print("Performing optimizer step")
                self.optimizer.step()
                running_loss += loss.item()
                # print("Performing validation loss...")
                # running_val_loss += self.validation_loss()
                # print("Finishing batch")

            # Compute validation loss once per epoch
            current_val_loss = self.validation_loss()
            running_val_loss += current_val_loss

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {running_loss/len(self.data_loader.train_loader)}, Val Loss: {running_val_loss/len(self.data_loader.val_loader)}")

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.save_model(epoch + 1, current_val_loss, is_best=True)

            # print("Updating cache...")
            self.data_loader.update_cache()
            # print( f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {running_loss/len(self.data_loader.train_loader)}")

        self.data_loader.shutdown_cache()

        

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        r2_metric = R2Score().to(self.device)
        with torch.no_grad():
            for batch_data in self.data_loader.test_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].float().to(self.device)
                predicted = self.model(images).flatten()

                loss = self.criterion(predicted, labels)
                total_loss += loss.item()
                r2_metric.update(predicted, labels)

        r2_score = r2_metric.compute()
        print(f'R^2 score of the network on the test images: {r2_score}')
        print(f"Test Loss: {total_loss/len(self.data_loader.test_loader)}")


    def save_model(self, epoch, val_loss, is_best=False):
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model_state, save_path)
        print(f"Model saved at {save_path}")

        if is_best:
            best_save_path = os.path.join(save_dir, "model_best.pth")
            torch.save(model_state, best_save_path)
            print(f"Best model saved at {best_save_path}")

    
    def load_model(self, checkpoint_path: str):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

        checkpoint: dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))

        print(f"Model and optimizer state loaded from '{checkpoint_path}'")
        return epoch, val_loss
