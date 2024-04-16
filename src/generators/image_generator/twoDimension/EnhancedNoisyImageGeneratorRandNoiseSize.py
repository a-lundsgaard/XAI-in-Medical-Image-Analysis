import cv2
import numpy as np
import math
from src.generators.utils.ImageGenerator import ImageGenerator;

class EnhancedNoisyImageGeneratorRandNoiseShape(ImageGenerator):
    def __init__(self, num_images, scalar=4, image_size=(256, 256), square_size=15, maxSize=100, minSize=5):
        self.scalar = scalar
        # Apply the scalar to image size, and shape size parameters
        scaled_image_size = (image_size[0] * scalar, image_size[1] * scalar)
        super().__init__(num_images, scaled_image_size, maxSize=maxSize * scalar, minSize=minSize * scalar)
        self.square_size = square_size * scalar

    def make_odd(self, size):
        """Ensure the kernel size is odd."""
        return size if size % 2 != 0 else size + 1

    def check_non_overlap(self, center1, size1, center2, size2):
        """Calculate the Euclidean distance to ensure non-overlap, including a scaled buffer."""
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance > (size1 + size2 + 10 * self.scalar)  # Scale the buffer as well

    def generate_images(self):
        gray_values = [50, 100, 150, 200]  # Different grayscale values for each shape
        for _ in range(self.num_images):
            self.square_size = np.random.randint(20 * self.scalar, 50 * self.scalar)  # Scale the random range for square size
            overlap = True
            while overlap:
                shapes_img = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
                noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0]), dtype=np.uint8)
                # Apply make_odd to ensure the kernel size is correct
                kernel_size = self.make_odd(5 * self.scalar)
                background = cv2.GaussianBlur(noise, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

                circle_radius = np.random.randint(self.minSize, self.maxSize)
                circle_center, square_center, triangle_center, ellipse_center = [
                    (np.random.randint(shape_size, self.image_size[0] - shape_size),
                     np.random.randint(shape_size, self.image_size[1] - shape_size)) 
                    for shape_size in (circle_radius, self.square_size, self.square_size // 2, self.square_size // 2)
                ]
                ellipse_axes = (self.square_size // 8, self.square_size // 4)
                ellipse_angle = np.random.randint(0, 360)

                overlap = not all(self.check_non_overlap(*pair) for pair in [
                    (circle_center, circle_radius, other_center, other_size)
                    for other_center, other_size in [
                        (square_center, self.square_size / math.sqrt(2)),
                        (triangle_center, self.square_size / 2),
                        (ellipse_center, max(ellipse_axes))
                    ]
                ])

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
                    cv2.fillPoly(shapes_img, [triangle_points], gray_values[2])
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
