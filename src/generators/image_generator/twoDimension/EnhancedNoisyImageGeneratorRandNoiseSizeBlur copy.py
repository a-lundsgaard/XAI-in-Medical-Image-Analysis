import cv2
import numpy as np
import math
from src.generators.utils.ImageGenerator import ImageGenerator

class EnhancedNoisyImageGeneratorRandNoiseShapeBlur(ImageGenerator):
    def __init__(self, num_images, image_size=(256, 256), square_size=15, maxSize=100, minSize=5):
        super().__init__(num_images, image_size, maxSize=maxSize, minSize=minSize)
        self.square_size = square_size

    def check_non_overlap(self, center1, size1, center2, size2):
        """Calculate the Euclidean distance to ensure non-overlap."""
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance > (size1 + size2 + 10)

    def generate_images(self):
        gray_values = [50, 100, 150, 200]  # Different grayscale values for each shape
        for _ in range(self.num_images):
            self.square_size = np.random.randint(20, 50)
            overlap = True
            while overlap:
                shapes_img = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
                mask = np.zeros_like(shapes_img)  # Mask for blurring edges
                noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0]), dtype=np.uint8)
                background = cv2.GaussianBlur(noise, (5, 5), cv2.BORDER_DEFAULT)

                # Define shape properties
                circle_radius = np.random.randint(self.minSize, self.maxSize)
                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))

                square_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                 np.random.randint(self.square_size, self.image_size[1] - self.square_size))

                triangle_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                   np.random.randint(self.square_size, self.image_size[1] - self.square_size))
                triangle_size = self.square_size / 2

                ellipse_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                  np.random.randint(self.square_size, self.image_size[1] - self.square_size))
                ellipse_axes = (self.square_size // 4, self.square_size // 2)
                ellipse_angle = np.random.randint(0, 360)

                # Draw shapes on mask
                cv2.circle(mask, circle_center, circle_radius, 255, -1)
                cv2.rectangle(mask, (square_center[0] - self.square_size // 2, square_center[1] - self.square_size // 2),
                              (square_center[0] + self.square_size // 2, square_center[1] + self.square_size // 2), 255, -1)
                triangle_points = np.array([
                    [triangle_center[0], triangle_center[1] - triangle_size],
                    [triangle_center[0] - triangle_size, triangle_center[1] + triangle_size],
                    [triangle_center[0] + triangle_size, triangle_center[1] + triangle_size]], dtype=np.int32)
                cv2.fillPoly(mask, [triangle_points], 255)
                cv2.ellipse(mask, ellipse_center, ellipse_axes, ellipse_angle, 0, 360, 255, -1)

                # Blur the mask to create blurred edges
                blurred_mask = cv2.GaussianBlur(mask, (21, 21), 10)

                # Apply the blurred mask with gray values to shapes_img
                for i, gray_value in enumerate(gray_values):
                    shapes_img[blurred_mask == 255] = gray_value

                angle = np.random.randint(0, 360)
                M = cv2.getRotationMatrix2D((self.image_size[0] / 2, self.image_size[1] / 2), angle, 1)
                rotated_shapes = cv2.warpAffine(shapes_img, M, (self.image_size[0], self.image_size[1]))

                # Drawing contours from blurred shapes on background
                contours, _ = cv2.findContours(rotated_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.drawContours(background, [cnt], 0, 255, -1)

                img = background
                self.images.append(img)
                self.labels.append(circle_radius)
