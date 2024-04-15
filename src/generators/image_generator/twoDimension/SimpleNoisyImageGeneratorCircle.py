import cv2
import numpy as np
from src.generators.utils.ImageGenerator import ImageGenerator;

class SimpleNoisyCircleImageGenerator(ImageGenerator):
    """
    Generates a set of noisy images with a single circle on a static blurred background,
    with fixed circle sizes (labels) in grayscale. The background blur does not rotate with the circle.
    """
    def __init__(self, num_images, image_size=(256, 256), maxSize=100):
        super().__init__(num_images, image_size, maxSize)
        # Assuming self.fixed_labels is defined elsewhere in the class or inherited.

    def generate_images(self):
        for _ in range(self.num_images):
            # Create a blank grayscale image for the circle
            shapes_img = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)

            # Create a noisy background and apply Gaussian blur
            noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0]), dtype=np.uint8)
            background = cv2.GaussianBlur(noise, (5, 5), cv2.BORDER_DEFAULT)

            # Set circle properties
            # circle_radius = np.random.choice(self.fixed_labels)
            circle_radius = np.random.randint(5, self.maxSize)
            circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                             np.random.randint(circle_radius, self.image_size[1] - circle_radius))

            # Draw the circle in white on the shapes image
            cv2.circle(shapes_img, circle_center, circle_radius, 255, -1)

            # Rotate the circle
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D((self.image_size[0] / 2, self.image_size[1] / 2), angle, 1)
            rotated_shapes = cv2.warpAffine(shapes_img, M, (self.image_size[0], self.image_size[1]))

            # Find contours in the rotated shapes image to copy the circle onto the background
            contours, _ = cv2.findContours(rotated_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(background, [cnt], 0, 255, -1)

            # The background now has the circle drawn onto it directly
            img = background

            # Add the grayscale image with the solid circle and non-rotated blur background, and its label to the list
            self.images.append(img)
            self.labels.append(circle_radius)
