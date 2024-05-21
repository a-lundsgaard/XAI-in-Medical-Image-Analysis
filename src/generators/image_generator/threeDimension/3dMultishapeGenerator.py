import numpy as np
import os
import torch
from skimage.draw import ellipsoid
from concurrent.futures import ProcessPoolExecutor


class ShapeGenerator3D:
    def __init__(self, num_volumes=3, shape=(256, 256, 256), center_deviation=0.1):
        self.output_dir = self.generate_path("datasets/artificial_data/3d_shapes/")
        self.num_volumes = num_volumes
        self.shape = shape
        self.center_deviation = center_deviation
        os.makedirs(self.output_dir, exist_ok=True)
        print("output dir: ", self.output_dir)
    
    def draw_sphere(self, volume, center, radius):
        ellip = ellipsoid(radius, radius, radius)
        rr, cc, zz = np.nonzero(ellip)
        rr = rr + center[0] - radius
        cc = cc + center[1] - radius
        zz = zz + center[2] - radius

        valid = (
            (rr >= 0) & (rr < volume.shape[0]) &
            (cc >= 0) & (cc < volume.shape[1]) &
            (zz >= 0) & (zz < volume.shape[2])
        )
        if not np.any(volume[rr[valid], cc[valid], zz[valid]]):
            volume[rr[valid], cc[valid], zz[valid]] = 255
            return True
        return False

    def draw_cube(self, volume, center, size):
        start = np.maximum(0, center - size // 2)
        end = np.minimum(volume.shape, center + size // 2 + 1)
        cube_volume = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        if not np.any(cube_volume):
            volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 255
            return True
        return False

    def draw_pyramid(self, volume, base_center, height):
        base_size = height * 2
        x, y, z = base_center
        for i in range(height):
            size = base_size - i * 2
            start = np.maximum(0, [x - size // 2, y - size // 2, z + i])
            end = np.minimum(volume.shape, [x + size // 2 + 1, y + size // 2 + 1, z + i + 1])
            if np.any(volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]):
                return False
        for i in range(height):
            size = base_size - i * 2
            start = np.maximum(0, [x - size // 2, y - size // 2, z + i])
            end = np.minimum(volume.shape, [x + size // 2 + 1, y + size // 2 + 1, z + i + 1])
            volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 255
        return True

    def draw_ellipsoid(self, volume, center, radii):
        ellip = ellipsoid(radii[0], radii[1], radii[2])
        rr, cc, zz = np.nonzero(ellip)
        rr = rr + center[0] - radii[0]
        cc = cc + center[1] - radii[1]
        zz = zz + center[2] - radii[2]

        valid = (
            (rr >= 0) & (rr < volume.shape[0]) &
            (cc >= 0) & (cc < volume.shape[1]) &
            (zz >= 0) & (zz < volume.shape[2])
        )
        if not np.any(volume[rr[valid], cc[valid], zz[valid]]):
            volume[rr[valid], cc[valid], zz[valid]] = 255
            return True
        return False

    def add_noise(self, volume, noise_level=50):
        noise = np.random.normal(0, noise_level, volume.shape)
        noisy_volume = np.clip(volume + noise, 0, 255).astype(np.uint8)
        return noisy_volume

    def generate_single_volume(self, vol):
        volume = np.zeros(self.shape, dtype=np.uint8)
        placed = False
        while not placed:
            volume.fill(0)
            center = np.array([self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2])
            
            # Attempt to place sphere
            for _ in range(100):
                sphere_radius = np.random.randint(10, 30)
                center_deviation = int(self.shape[0] * self.center_deviation) # deviate the center by 10% of the volume size
                if self.draw_sphere(volume, center + np.random.randint(-center_deviation, center_deviation, size=3), sphere_radius):
                    break
            
            # Attempt to place cube
            for _ in range(100):
                size = np.random.randint(20, 50)
                if self.draw_cube(volume, center + np.random.randint(-50, 50, size=3), size):
                    break
            
            # Attempt to place pyramid
            for _ in range(100):
                height = np.random.randint(10, 30)
                if self.draw_pyramid(volume, center + np.random.randint(-50, 50, size=3), height):
                    break
            
            # Attempt to place ellipsoid
            for _ in range(100):
                radii = np.random.randint(10, 30, size=3)
                if self.draw_ellipsoid(volume, center + np.random.randint(-50, 50, size=3), radii):
                    break

            placed = True

        # Add background noise
        volume = self.add_noise(volume, noise_level=10)
        
        # Convert the volume to a PyTorch tensor and save it
        tensor_volume = torch.tensor(volume)
        torch.save(tensor_volume, os.path.join(self.output_dir, f'volume_{vol}_label_{sphere_radius}.pt'))

    def generate_images(self):
        with ProcessPoolExecutor() as executor:
            executor.map(self.generate_single_volume, range(self.num_volumes))

    def generate_path(self, workspace_path):
        file_path = os.path.dirname(os.path.abspath(__file__))
        root_path = file_path.split("src")[0]
        return root_path + workspace_path


if __name__ == '__main__':
    generator = ShapeGenerator3D(num_volumes=50, center_deviation=0.1)
    generator.generate_images()