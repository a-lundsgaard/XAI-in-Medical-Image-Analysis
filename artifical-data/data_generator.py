import cv2
import numpy as np
import math
import os

def generate_noisy_images(num_images, image_size=(256, 256), square_size=50, fixed_labels=[20, 40, 60, 80, 100]):
    """
    Generates a set of images with a square and a circle, ensuring they do not overlap, with fixed circle sizes (labels).
    The images include random noise and blurred lines.

    Args:
    - num_images: Number of images to generate.
    - image_size: Size of the image (width, height).
    - square_size: Side length of the square.
    - fixed_labels: List of fixed circle sizes to use as labels.

    Returns:
    - A list of tuples, each containing an image and its label (one of the fixed circle sizes).
    """
    images = []
    labels = []

    for _ in range(num_images):
        while True:
            # Create a blank image
            img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

            # Add random noise to the background
            noise = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, noise, 0.5, 0)

            # Set circle properties (selecting randomly from the fixed labels)
            circle_radius = np.random.choice(fixed_labels)
            circle_center = (np.random.randint(circle_radius, image_size[0] - circle_radius), 
                             np.random.randint(circle_radius, image_size[1] - circle_radius))

            # Set square properties
            square_center = (np.random.randint(square_size, image_size[0] - square_size), 
                             np.random.randint(square_size, image_size[1] - square_size))

            # Check distance to ensure no overlap
            distance = math.sqrt((square_center[0] - circle_center[0]) ** 2 + (square_center[1] - circle_center[1]) ** 2)
            min_distance = circle_radius + (math.sqrt(2) * square_size) / 2  # Circle radius + half the diagonal of the square

            if distance > min_distance:
                break  # No overlap, can proceed

        # Draw the square and circle
        cv2.rectangle(img, (square_center[0] - square_size // 2, square_center[1] - square_size // 2),
                      (square_center[0] + square_size // 2, square_center[1] + square_size // 2), (255, 255, 255), -1)
        cv2.circle(img, circle_center, circle_radius, (255, 255, 255), -1)

        # Apply blurring to create blurred lines
        img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

        # Random rotation
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((image_size[0] / 2, image_size[1] / 2), angle, 1)
        rotated_img = cv2.warpAffine(img, M, (image_size[0], image_size[1]))

        # Add the image and its label (fixed circle size) to the list
        images.append(rotated_img)
        labels.append(circle_radius)

    return images, labels

# Generate noisy images
output_dir = 'noisy_generated_images'
os.makedirs(output_dir, exist_ok=True)

noisy_images, noisy_labels = generate_noisy_images(100)

for i, (img, label) in enumerate(zip(noisy_images, noisy_labels)):
    filename = os.path.join(output_dir, f'noisy_image_{i+1}_label_{label}.png')
    cv2.imwrite(filename, img)
