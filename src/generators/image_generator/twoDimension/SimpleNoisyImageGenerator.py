import cv2
import numpy as np
import math
# from generators.utils.ImageGenerator import ImageGenerator;
from utils.ImageGenerator import ImageGenerator


class SimpleNoisyImageGenerator(ImageGenerator):
    """
    Generates a set of noisy images with a square and a circle, ensuring they do not overlap,
    with fixed circle sizes (labels).
    """
    def __init__(self, num_images, image_size=(256, 256), square_size=50):
        super().__init__(num_images, image_size)
        self.square_size = square_size
        # self.fixed_labels = fixed_labels

    # def get_model_name(self):
    #     return self.__class__.__name__

    def generate_images(self):
        for _ in range(self.num_images):
            while True:
                # Create a blank image
                img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

                # Add random noise to the background
                noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
                img = cv2.addWeighted(img, 0.5, noise, 0.5, 0)

                # Set circle properties
                circle_radius = np.random.choice(self.fixed_labels)
                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius), 
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))

                # Set square properties
                square_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size), 
                                 np.random.randint(self.square_size, self.image_size[1] - self.square_size))

                # Check distance to ensure no overlap
                distance = math.sqrt((square_center[0] - circle_center[0]) ** 2 + (square_center[1] - circle_center[1]) ** 2)
                min_distance = circle_radius + (math.sqrt(2) * self.square_size) / 2  # Circle radius + half the diagonal of the square

                if distance > min_distance:
                    break  # No overlap, can proceed

            # Draw the square and circle
            cv2.rectangle(img, (square_center[0] - self.square_size // 2, square_center[1] - self.square_size // 2),
                          (square_center[0] + self.square_size // 2, square_center[1] + self.square_size // 2), (255, 255, 255), -1)
            cv2.circle(img, circle_center, circle_radius, (255, 255, 255), -1)

            # Apply blurring to create blurred lines
            img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

            # Random rotation
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D((self.image_size[0] / 2, self.image_size[1] / 2), angle, 1)
            img = cv2.warpAffine(img, M, (self.image_size[0], self.image_size[1]))

            # Add the image and its label to the list
            self.images.append(img)
            self.labels.append(circle_radius)
