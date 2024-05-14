from monai.networks.nets import ResNet, resnet18, resnet34, resnet50
import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torcheval.metrics import R2Score
from torch import Tensor


class MedicalResNetModel:
    def __init__(self,
                 num_epochs,
                 data_loader: NiftiDataLoader,
                 learning_rate=0.01,
                 weight_decay=None,
                 dropout_rate=None,
                 depth=18,
                 spatial_dims=3,
                 pretrained=True
                 ):

        self.spacial_dims = spatial_dims
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_loader: NiftiDataLoader = data_loader

        # check what spatial dimensions the first training image is and set the model to that
        if self.data_loader.train_loader:
            for batch_data in self.data_loader.train_loader:
                images = batch_data["image"]
                break
            print("Image spatial dimensions: ", len(images.shape) - 2)
            self.spacial_dims = len(images.shape) - 2

        # set the n_input_channels based on the number of channels in the first image
        n_input_channels = images.shape[1]
        print("Number of input channels: ", n_input_channels)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        resnet: ResNet = None
        if depth == 18:
            resnet = resnet18
        elif depth == 34:
            resnet = resnet34
        elif depth == 50:
            resnet = resnet50
        else:
            raise ValueError(
                "Unsupported depth for ResNet. Choose from 18, 34, 50.")

        # Define the model
        self.model: ResNet = resnet(
            spatial_dims=self.spacial_dims,
            n_input_channels=n_input_channels,
            pretrained=pretrained,
            # dropout_rate=dropout_rate,
        )
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Data loaders

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
