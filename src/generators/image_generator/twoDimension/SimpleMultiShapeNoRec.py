import cv2
import numpy as np
import math
from src.generators.utils.ImageGenerator import ImageGenerator

class SimpleMultiShapeNoRec(ImageGenerator):
    def __init__(self, num_images, scalar=1, image_size=(256, 256), maxSize=100, minSize=5):
        self.scalar = scalar
        # Apply the scalar to image size, and shape size parameters
        scaled_image_size = (image_size[0] * scalar, image_size[1] * scalar)
        super().__init__(num_images, scaled_image_size, maxSize=maxSize * scalar, minSize=minSize * scalar)

    def check_non_overlap(self, center1, size1, center2, size2):
        """Calculate the Euclidean distance to ensure non-overlap, including a scaled buffer."""
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance > (size1 + size2 + 10 * self.scalar)  # Scale the buffer as well

    def generate_images(self):
        gray_values = [50, 100, 150, 200]  # Different grayscale values for each shape
        for _ in range(self.num_images):
            overlap = True
            while overlap:
                shapes_img = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
                noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0]), dtype=np.uint8)
                background = cv2.GaussianBlur(noise, (5 * self.scalar, 5 * self.scalar), cv2.BORDER_DEFAULT)

                circle_radius = np.random.randint(self.minSize, self.maxSize)
                triangle_size = np.random.randint(30, 100)
                ellipse_axes = (triangle_size // 8, triangle_size // 4)
                ellipse_angle = np.random.randint(0, 360)

                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))
                triangle_center = (np.random.randint(triangle_size, self.image_size[0] - triangle_size),
                                   np.random.randint(triangle_size, self.image_size[1] - triangle_size))
                ellipse_center = (np.random.randint(ellipse_axes[0], self.image_size[0] - ellipse_axes[0]),
                                  np.random.randint(ellipse_axes[1], self.image_size[1] - ellipse_axes[1]))

                overlap = not all(self.check_non_overlap(*pair) for pair in [
                    (circle_center, circle_radius, triangle_center, triangle_size / 2),
                    (circle_center, circle_radius, ellipse_center, max(ellipse_axes)),
                    (ellipse_center, max(ellipse_axes), triangle_center, triangle_size / 2)
                ])

                if not overlap:
                    # Draw shapes
                    cv2.circle(shapes_img, circle_center, circle_radius, gray_values[1], -1)
                    triangle_points = np.array([
                        [triangle_center[0], triangle_center[1] - triangle_size // 2],
                        [triangle_center[0] - triangle_size // 2, triangle_center[1] + triangle_size // 2],
                        [triangle_center[0] + triangle_size // 2, triangle_center[1] + triangle_size // 2]
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
