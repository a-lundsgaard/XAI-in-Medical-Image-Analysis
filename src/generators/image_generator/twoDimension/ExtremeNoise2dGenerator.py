import cv2
import numpy as np
import math
# Assuming ImageGenerator is correctly defined in your project structure
from src.generators.utils.ImageGenerator import ImageGenerator;


class ExtremeNoise2dGenerator(ImageGenerator):
    """
    Generates a set of extremely noisy images with varying squares and fixed-size circles,
    ensuring they do not overlap, with variable grayscale colors, blending into the extremely noisy background.
    """
    def __init__(self, num_images, image_size=(256, 256), square_min_size=20, square_max_size=100, blur_intensity=(25, 25)):
        super().__init__(num_images, image_size)
        self.square_min_size = square_min_size
        self.square_max_size = square_max_size
        self.blur_intensity = blur_intensity

    def add_extreme_noise(self, image):
        noise = np.random.normal(loc=128, scale=128, size=image.shape)
        noisy_image = np.clip(noise, 0, 255).astype(np.uint8)
        return noisy_image

    def generate_images(self):
        for _ in range(self.num_images):
            while True:
                # Create a noisy background for a grayscale image
                img = self.add_extreme_noise(np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8))

                # Randomize square properties and choose fixed circle size
                square_size = np.random.randint(self.square_min_size, self.square_max_size)
                # circle_radius = np.random.choice(self.fixed_labels)
                circle_radius = np.random.randint(5, 100)
                shape_color = np.random.randint(50, 206)  # Grayscale value to blend with background

                # Random positions ensuring they do not exceed image boundaries
                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))
                square_center = (np.random.randint(square_size, self.image_size[0] - square_size),
                                 np.random.randint(square_size, self.image_size[1] - square_size))

                # Ensure no overlap between square and circle
                distance = math.sqrt((square_center[0] - circle_center[0])**2 + (square_center[1] - circle_center[1])**2)
                min_distance = circle_radius + math.sqrt(2) * square_size / 2

                if distance > min_distance:
                    break  # No overlap, can proceed

            # Draw the square and circle with the same grayscale value on the noisy background
            cv2.rectangle(img, (square_center[0] - square_size // 2, square_center[1] - square_size // 2),
                          (square_center[0] + square_size // 2, square_center[1] + square_size // 2), shape_color, -1)
            cv2.circle(img, circle_center, circle_radius, shape_color, -1)

            # Apply heavier blurring to the entire image to increase noise
            img = cv2.GaussianBlur(img, self.blur_intensity, cv2.BORDER_DEFAULT)

            # Add the image and its label (circle's fixed size) to the list
            self.images.append(img)
            self.labels.append(circle_radius)
