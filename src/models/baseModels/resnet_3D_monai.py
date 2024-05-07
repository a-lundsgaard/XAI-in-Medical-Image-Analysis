from monai.networks.nets import ResNet, resnet18, resnet34, resnet50
import torch
import torch.nn as nn
from src.dataLoaders.NiftiDataLoader2 import NiftiDataLoader
from torcheval.metrics import R2Score
from torch import Tensor


class MedicalResNetModel:
    def __init__(self,
                 num_epochs,
                 learning_rate=0.01,
                 data_loader=NiftiDataLoader,
                 weight_decay=None,
                 dropout_rate=None,
                 depth=18,
                 spatial_dims=3
                 ):

        self.spacial_dims = spatial_dims
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_loader: NiftiDataLoader = data_loader

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
            n_input_channels=1,
            # dropout_rate=dropout_rate,
        )
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)  # Output layer for regression.

        self.model.to(self.device)


        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Data loaders

    def validation_loss(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.data_loader.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).flatten()
                loss = self.criterion(outputs, labels.float())
                return loss.item()

    def train(self):

        self.model.train()
        for epoch in range(self.num_epochs):
            # self.model.train()
            running_loss = 0.0
            running_val_loss = 0.0

            for batch_data in self.data_loader.train_loader:
                # images, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)

                # print(f"Batch data: {batch_data}")

                # print(f"Batch data image: {batch_data['image']}")

                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].float().to(self.device)
                # labels = batch_data["label"].to(
                #     self.device)  # Ensure labels are float32

                # images = batch_data["image"]
                # labels = batch_data["label"] # Ensure labels are float32

                print(f"Images shape: {images.shape}")
                print(f"Labels: {labels}")

                self.optimizer.zero_grad()
                outputs = self.model(images)
                print(f"Outputs shape: {outputs}")

                loss: Tensor = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # running_val_loss += self.validation_loss()

            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {running_loss/len(self.data_loader.train_loader)}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        r2_metric = R2Score().to(self.device)
        with torch.no_grad():
            for batch_data in self.data_loader.test_loader:
                images, labels = batch_data["image"].to(
                    self.device), batch_data["label"].to(self.device)
                outputs = self.model(images)
                predicted = outputs.flatten()

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                r2_metric.update(predicted, labels)

        r2_score = r2_metric.compute()
        print(f'R^2 score of the network on the test images: {r2_score}')
        print(f"Test Loss: {total_loss/len(self.data_loader.test_loader)}")
