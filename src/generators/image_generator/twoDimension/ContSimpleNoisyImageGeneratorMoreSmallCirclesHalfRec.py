import cv2
import numpy as np
import math
from src.generators.utils.ImageGenerator import ImageGenerator

class ContSimpleNoisyImageGeneratorMoreSmallCirclesHalfRec1(ImageGenerator):
    def __init__(self, num_images, image_size=(256, 256), square_size=50, minSize=5, maxSize=100):
        super().__init__(num_images, image_size)
        self.square_size = square_size
        self.max_size = maxSize
        self.min_size = minSize

    def generate_images(self):
        np.random.seed(42)
        for _ in range(self.num_images):
            while True:
                shapes_img = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
                noise = np.random.randint(0, 256, (self.image_size[1], self.image_size[0]), dtype=np.uint8)
                background = cv2.GaussianBlur(noise, (5, 5), cv2.BORDER_DEFAULT)

                circle_radius = np.random.randint(self.min_size, self.max_size)
                p = 0.1
                scale = 0.5
                should_reduce_radius = np.random.choice([0, 1], p=[p, 1-p])
                if should_reduce_radius:
                    circle_radius = int(circle_radius * scale)

                circle_center = (np.random.randint(circle_radius, self.image_size[0] - circle_radius),
                                 np.random.randint(circle_radius, self.image_size[1] - circle_radius))

                if np.random.rand() > 0.5:  # Draw the square only half of the time
                    square_center = (np.random.randint(self.square_size, self.image_size[0] - self.square_size),
                                     np.random.randint(self.square_size, self.image_size[1] - self.square_size))

                    distance = math.sqrt((square_center[0] - circle_center[0]) ** 2 + (square_center[1] - circle_center[1]) ** 2)
                    min_distance = circle_radius + (math.sqrt(2) * self.square_size) / 2

                    if distance > min_distance:
                        cv2.rectangle(shapes_img, (square_center[0] - self.square_size // 2, square_center[1] - self.square_size // 2),
                                      (square_center[0] + self.square_size // 2, square_center[1] + self.square_size // 2), 255, -1)
                        break
                else:
                    break

            cv2.circle(shapes_img, circle_center, circle_radius, 255, -1)
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D((self.image_size[0] / 2, self.image_size[1] / 2), angle, 1)
            rotated_shapes = cv2.warpAffine(shapes_img, M, (self.image_size[0], self.image_size[1]))

            contours, _ = cv2.findContours(rotated_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(background, [cnt], 0, 255, -1)

            img = background
            self.images.append(img)
            self.labels.append(circle_radius)
