import os
from abc import ABC, abstractmethod
import cv2

class ImageGenerator(ABC):
    """
    Abstract class for generating images.
    """
    def __init__(self, num_images, image_size):
        self.num_images = num_images
        self.image_size = image_size
        self.fixed_labels = [20, 40, 60, 80, 100]
        self.images = []
        self.labels = []

    @abstractmethod
    def generate_images(self):
        """
        Generates a set of images. This method needs to be implemented by subclasses.
        """
        pass

    def get_model_name(self):
        """
        Generates a set of images. This method needs to be implemented by subclasses.
        """
        return self.__class__.__name__

    def save_images(self, output_dir):
        """
        Saves the generated images to the specified directory.

        Args:
        - output_dir: Directory where the images will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            filename = os.path.join(output_dir, f'image_{i + 1}_label_{label}.png')
            cv2.imwrite(filename, img)

