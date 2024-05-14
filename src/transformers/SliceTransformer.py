from monai.transforms import MapTransform
from monai.config import KeysCollection
import numpy as np

class SliceAggregateTransform(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img_data = d[key]
            # print(f"Image data shape: {img_data.shape}")  # Debugging: print the shape of the image data
            
            # Remove the channel dimension if it exists
            if img_data.ndim == 4 and img_data.shape[0] == 1:
                img_data = img_data[0]
            
            combined_images = self.slice_and_aggregate(img_data)
            d[key] = combined_images
        return d
    
    def slice_and_aggregate(self, img_data):
        """
        Slices the 3D image from all three directions and aggregates them into three separate 3-channel images.
        """
        # Ensure the image data is 3D (mri data)
        if img_data.ndim != 3:
            raise ValueError(f"Expected 3D image data, but got {img_data.ndim}D data.")

        depth, height, width = img_data.shape

        # Validate dimensions to avoid index errors
        if depth < 3 or height < 3 or width < 3:
            raise ValueError(f"Image data is too small to slice: {img_data.shape}")

        # Slicing from different directions
        axial_slices = [img_data[:, :, width//2 - 1], 
                        img_data[:, :, width//2], 
                        img_data[:, :, width//2 + 1]]

        coronal_slices = [img_data[:, height//2 - 1, :], 
                          img_data[:, height//2, :], 
                          img_data[:, height//2 + 1, :]]

        sagittal_slices = [img_data[depth//2 - 1, :, :], 
                           img_data[depth//2, :, :], 
                           img_data[depth//2 + 1, :, :]]

        # Stack each set of slices into separate channels
        axial_image = np.stack(axial_slices, axis=0)
        coronal_image = np.stack(coronal_slices, axis=0)
        sagittal_image = np.stack(sagittal_slices, axis=0)

        # Combine the three directional images into a single multichannel image
        # combined = np.array([axial_image, coronal_image, sagittal_image])
        combined = np.concatenate((axial_image, coronal_image, sagittal_image), axis=0)
        return combined
