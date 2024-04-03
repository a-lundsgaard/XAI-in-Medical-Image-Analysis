from src.generators.utils.ImageGenerator import ImageGenerator;
import shutil
import os

class GenerateImages:
    def __init__(self, generator: ImageGenerator):
        self.generator = generator
        self.output_dir = os.path.abspath("../datasets/artificial_data/")        
        self.suffix = "_images"

    def checkIfDataExists(self):
        return os.path.exists(self.generate_path())

    def generateDataSet(self):
        self.save_images()
        return self.generator.get_model_name()

    def generate_path(self):
        return os.path.join(self.output_dir, self.generator.get_model_name())

    def generate_images(self):
        self.generator.generate_images()

    def save_images(self):
        # delete the directory if it exists
        if self.checkIfDataExists():
            shutil.rmtree(self.generate_path())  # This will remove the directory and all its contents
            
        self.generator.generate_images()
        self.generator.save_images(self.generate_path())

    
