from utils.ImageGenerator import ImageGenerator
import os

class GenerateImages:
    def __init__(self, generator: ImageGenerator):
        self.generator = generator
        self.output_dir = "../../datasets/artificial_data/"
        self.suffix = "_images"

    def generate_path(self):
        return os.path.join(self.output_dir, self.generator.get_model_name() + self.suffix)

    def generate_images(self):
        self.generator.generate_images()

    def save_images(self):
        self.generator.save_images(self.generate_path())
