import cv2
import numpy as np
import math
from src.generators.utils.ImageGenerator import ImageGenerator

class ExtremeNoise2dGeneratorMultiShape(ImageGenerator):
    """
    Generates a set of extremely noisy images with varying squares, circles, ellipses, and triangles,
    ensuring they do not overlap, with variable grayscale colors, blending into the extremely noisy background.
    """
    def __init__(self, num_images, image_size=(256, 256), minSize=20, maxSize=100, blur_intensity=(25, 25)):
        super().__init__(num_images, image_size, maxSize=maxSize, minSize=minSize)
        self.square_min_size = minSize
        self.square_max_size = maxSize
        self.blur_intensity = blur_intensity

    def add_extreme_noise(self, image):
        noise = np.random.normal(loc=128, scale=128, size=image.shape)
        noisy_image = np.clip(noise, 0, 255).astype(np.uint8)
        return noisy_image

    def check_non_overlap(self, center1, size1, center2, size2):
        """Calculate the Euclidean distance to ensure non-overlap."""
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance > (size1 + size2 + 10)  # Adding a small buffer to ensure clear non-overlap

    def generate_images(self):
        for _ in range(self.num_images):
            while True:
                # Create a noisy background for a grayscale image
                img = self.add_extreme_noise(np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8))

                # Randomize properties
                square_size = np.random.randint(self.square_min_size, self.square_max_size)
                circle_radius = np.random.randint(5, 30)  # Random fixed circle size
                ellipse_size = (np.random.randint(10, 50), np.random.randint(10, 50))  # width, height of ellipse
                triangle_size = np.random.randint(10, 50)  # size of triangle
                shape_color = np.random.randint(50, 206)  # Grayscale value to blend with background

                # Random positions ensuring they do not exceed image boundaries
                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))
                square_center = (np.random.randint(square_size, self.image_size[0] - square_size),
                                 np.random.randint(square_size, self.image_size[1] - square_size))
                ellipse_center = (np.random.randint(ellipse_size[0], self.image_size[0] - ellipse_size[0]),
                                  np.random.randint(ellipse_size[1], self.image_size[1] - ellipse_size[1]))
                triangle_center = (np.random.randint(triangle_size, self.image_size[0] - triangle_size),
                                   np.random.randint(triangle_size, self.image_size[1] - triangle_size))

                # Ensure no overlap among shapes
                if (self.check_non_overlap(circle_center, circle_radius, square_center, square_size // 2) and
                    self.check_non_overlap(circle_center, circle_radius, ellipse_center, max(ellipse_size)) and
                    self.check_non_overlap(circle_center, circle_radius, triangle_center, triangle_size) and
                    self.check_non_overlap(square_center, square_size // 2, ellipse_center, max(ellipse_size)) and
                    self.check_non_overlap(square_center, square_size // 2, triangle_center, triangle_size) and
                    self.check_non_overlap(ellipse_center, max(ellipse_size), triangle_center, triangle_size)):
                    break

            # Draw the shapes
            cv2.rectangle(img, (square_center[0] - square_size // 2, square_center[1] - square_size // 2),
                          (square_center[0] + square_size // 2, square_center[1] + square_size // 2), shape_color, -1)
            cv2.circle(img, circle_center, circle_radius, shape_color, -1)
            cv2.ellipse(img, ellipse_center, ellipse_size, 0, 0, 360, shape_color, -1)
            triangle_points = np.array([
                [triangle_center[0], triangle_center[1] - triangle_size],
                [triangle_center[0] - triangle_size, triangle_center[1] + triangle_size],
                [triangle_center[0] + triangle_size, triangle_center[1] + triangle_size]], dtype=np.int32)
            cv2.fillPoly(img, [triangle_points], shape_color)

            # Apply heavier blurring to the entire image to increase noise
            img = cv2.GaussianBlur(img, self.blur_intensity, cv2.BORDER_DEFAULT)

            # Add the image and its label (circle's fixed size) to the list
            self.images.append(img)
            self.labels.append(circle_radius)
