import cv2
import numpy as np
import math
from utils.ImageGenerator import ImageGenerator; # Adjust according to your file structure

class SimpleNoisyImageGenerator(ImageGenerator):
    """
    Generates a set of noisy images with a square and a circle on a static blurred background,
    ensuring they do not overlap, with fixed circle sizes (labels) in grayscale. 
    The shapes are not semi-transparent and the background blur does not rotate with the shapes.
    """
    def __init__(self, num_images, image_size=(256, 256), square_size=50):
        super().__init__(num_images, image_size)
        self.square_size = square_size
        # Assuming self.fixed_labels is defined elsewhere in the class or inherited.

    def generate_images(self):
        for _ in range(self.num_images):
            while True:
                # Create a blank grayscale image for shapes
                shapes_img = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)

                # Create a noisy background and apply Gaussian blur
                noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0]), dtype=np.uint8)
                background = cv2.GaussianBlur(noise, (5, 5), cv2.BORDER_DEFAULT)

                # Set circle properties
                circle_radius = np.random.choice(self.fixed_labels)
                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))

                # Set square properties
                square_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                 np.random.randint(self.square_size, self.image_size[1] - self.square_size))

                # Check distance to ensure no overlap
                distance = math.sqrt((square_center[0] - circle_center[0]) ** 2 + (square_center[1] - circle_center[1]) ** 2)
                min_distance = circle_radius + (math.sqrt(2) * self.square_size) / 2

                if distance > min_distance:
                    break

            # Draw the square and circle in white on the shapes image
            cv2.rectangle(shapes_img, (square_center[0] - self.square_size // 2, square_center[1] - self.square_size // 2),
                          (square_center[0] + self.square_size // 2, square_center[1] + self.square_size // 2), 255, -1)
            cv2.circle(shapes_img, circle_center, circle_radius, 255, -1)

            # Rotate shapes
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D((self.image_size[0] / 2, self.image_size[1] / 2), angle, 1)
            rotated_shapes = cv2.warpAffine(shapes_img, M, (self.image_size[0], self.image_size[1]))

            # Find contours in the rotated shapes image to copy shapes onto the background
            contours, _ = cv2.findContours(rotated_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(background, [cnt], 0, 255, -1)

            # The background now has the shapes drawn onto it directly
            img = background

            # Add the grayscale image with solid shapes and non-rotated blur background, and its label to the list
            self.images.append(img)
            self.labels.append(circle_radius)
