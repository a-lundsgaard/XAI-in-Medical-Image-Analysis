import cv2
import numpy as np
import math
from src.generators.utils.ImageGenerator import ImageGenerator;

class EnhancedNoisyImageGeneratorRandNoiseShape(ImageGenerator):
    def __init__(self, num_images, image_size=(256, 256), square_size=15, maxSize=100, minSize=5):
        super().__init__(num_images, image_size, maxSize=maxSize, minSize=minSize)
        self.square_size = square_size
        # Assuming self.fixed_labels is defined elsewhere in the class or inherited.

    def check_non_overlap(self, center1, size1, center2, size2):
        """Calculate the Euclidean distance to ensure non-overlap."""
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        # Adjusting size sum to be more appropriate for geometric shapes
        return distance > (size1 + size2 + 10)  # Adding a small buffer to ensure clear non-overlap

    def generate_images(self):
        gray_values = [50, 100, 150, 200]  # Different grayscale values for each shape
        for _ in range(self.num_images):
            self.square_size = np.random.randint(20, 50)
            overlap = True
            while overlap:
                #print("Square size:", self.square_size)
                shapes_img = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
                noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0]), dtype=np.uint8)
                background = cv2.GaussianBlur(noise, (5, 5), cv2.BORDER_DEFAULT)

                circle_radius = np.random.randint(self.minSize, self.maxSize)
                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))

                square_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                 np.random.randint(self.square_size, self.image_size[1] - self.square_size))

                triangle_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                   np.random.randint(self.square_size, self.image_size[1] - self.square_size))
                triangle_size = self.square_size / 2  # Approximate 'radius' for distance checks

                ellipse_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                  np.random.randint(self.square_size, self.image_size[1] - self.square_size))
                ellipse_axes = (self.square_size // 4, self.square_size // 2)
                ellipse_angle = np.random.randint(0, 360)

                # Check distances between each pair of shapes
                if (self.check_non_overlap(circle_center, circle_radius, square_center, self.square_size / math.sqrt(2)) and
                    self.check_non_overlap(circle_center, circle_radius, triangle_center, triangle_size) and
                    self.check_non_overlap(circle_center, circle_radius, ellipse_center, max(ellipse_axes)) and
                    self.check_non_overlap(square_center, self.square_size / math.sqrt(2), triangle_center, triangle_size) and
                    self.check_non_overlap(square_center, self.square_size / math.sqrt(2), ellipse_center, max(ellipse_axes)) and
                    self.check_non_overlap(triangle_center, triangle_size, ellipse_center, max(ellipse_axes))):
                    overlap = False

                if not overlap:
                    # Draw shapes
                    cv2.rectangle(shapes_img, (square_center[0] - self.square_size // 2, square_center[1] - self.square_size // 2),
                                  (square_center[0] + self.square_size // 2, square_center[1] + self.square_size // 2), gray_values[0], -1)
                    cv2.circle(shapes_img, circle_center, circle_radius, gray_values[1], -1)
                    triangle_points = np.array([
                        [triangle_center[0], triangle_center[1] - self.square_size // 2],
                        [triangle_center[0] - self.square_size // 2, triangle_center[1] + self.square_size // 2],
                        [triangle_center[0] + self.square_size // 2, triangle_center[1] + self.square_size // 2]
                    ], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(shapes_img, [triangle_points], isClosed=True, color=gray_values[2], thickness=1)
                    cv2.ellipse(shapes_img, ellipse_center, ellipse_axes, ellipse_angle, 0, 360, gray_values[3], -1)

                    angle = np.random.randint(0, 360)
                    M = cv2.getRotationMatrix2D((self.image_size[0] / 2, self.image_size[1] / 2), angle, 1)
                    rotated_shapes = cv2.warpAffine(shapes_img, M, (self.image_size[0], self.image_size[1]))

                    contours, _ = cv2.findContours(rotated_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cv2.drawContours(background, [cnt], 0, 255, -1)

                    img = background
                    self.images.append(img)
                    self.labels.append(circle_radius)
