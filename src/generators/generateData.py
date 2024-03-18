from utils.GenerateImages import GenerateImages
from image_generator.twoDimension.SimpleNoisyImageGenerator import SimpleNoisyImageGenerator
from image_generator.twoDimension.ExtremeNoise2dGenerator import ExtremeNoise2dGenerator

# generate images
generator = GenerateImages(SimpleNoisyImageGenerator(num_images=1))

# generate images
# generator.generate_images()
# generator.save_images()

extreme_generator = GenerateImages(ExtremeNoise2dGenerator(num_images=20))
extreme_generator.generate_images()
extreme_generator.save_images()