# from monai.networks.nets import ResNet, resnet18, resnet34, resnet50
import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torcheval.metrics import R2Score
from torch import Tensor

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
                 ):

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_loader: NiftiDataLoader = data_loader
        self.model: Any = None
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



        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.set_model()
        self.load_weights()

        num_features = self.model.fc.in_features
        if dropout_rate:
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 1)
            )
        else:
            self.model.fc = nn.Linear(num_features, 1)  # Output layer for regression.

        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # def load_weights(self):
    #     pretrained_weights_path=f"../src/models/weights/resnet_{self.depth}_23dataset.pth"
    #     if self.pretrained and pretrained_weights_path is not None:
    #         state = torch.load(pretrained_weights_path, map_location=self.device)

    #         # print("Loading pretrained weights from: ", pretrained_weights_path)
    #         # print shape of state_dict
    #         # print(f"State dict shape: {state["state_dict"]}")
    #         # Handle state dictionary key adjustment
    #         if "state_dict" in state:
    #             state = state["state_dict"]
    #             # print("State dict key adjustment", state.keys())
    #         self.model.load_state_dict(state)

    def load_weights(self):
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

    # def load_weights(self):
        
    #     if self.pretrained and self.pretrained_weights_path is not None:
    #         state = torch.load(self.pretrained_weights_path, map_location=self.device)

    #         # Handle state dictionary key adjustment
    #         if "state_dict" in state:
    #             state = state["state_dict"]

    #         # Strip the `module.` prefix
    #         new_state_dict = {}
    #         for k, v in state.items():
    #             if k.startswith("module."):
    #                 new_state_dict[k[7:]] = v  # Remove the `module.` prefix
    #             else:
    #                 new_state_dict[k] = v

    #         print("State dict key adjustment", new_state_dict.keys())
    #         self.model.load_state_dict(new_state_dict)

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
        print("Is cuda available: ", torch.cuda.is_available())

        for epoch in range(self.num_epochs):
            # print("Setting model to train mode")
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
                running_val_loss += self.validation_loss()
                # print("Finishing batch")

            self.data_loader.update_cache()
            # print( f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {running_loss/len(self.data_loader.train_loader)}")
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {running_loss/len(self.data_loader.train_loader)}, Val Loss: {running_val_loss/len(self.data_loader.val_loader)}")

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
