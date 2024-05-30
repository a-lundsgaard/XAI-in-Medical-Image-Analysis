from monai.transforms import MapTransform
from monai.config import KeysCollection
import numpy as np

class SliceAggregateTransform(MapTransform):
    def __init__(self, keys: KeysCollection, slices_from_each_view=3, allow_missing_keys: bool = False, views=['coronal', 'sagittal', 'axial'], center_deviation=0.05):
        self.slices_from_each_view = slices_from_each_view
        self.views = views
        self.center_deviation = center_deviation
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img_data = d[key]
            # print(f"Image data shape: {img_data.shape}")  # Debugging: print the shape of the image data
            
            # Remove the channel dimension if it exists
            if img_data.ndim == 4 and img_data.shape[0] == 1:
                img_data = img_data[0]
            
            # check if slices_from_each_view is 1 or 3 else raise error
            if self.slices_from_each_view not in [1, 3]:
                raise ValueError(f"Expected slices_from_each_view to be 1 or 3, but got {self.slices_from_each_view}.")
            
            combined_images = self.slice_and_aggregate_3_channel(img_data) if self.slices_from_each_view == 1 else self.slice_and_aggregate(img_data)
            d[key] = combined_images
        return d
    
    def slice_and_aggregate_3_channel(self, img_data):
        """
        Slices the 3D image from all three directions and aggregates them into a single 3-channel image.
        """
        # Ensure the image data is 3D (mri data)
        if img_data.ndim != 3:
            raise ValueError(f"Expected 3D image data, but got {img_data.ndim}D data.")

        depth, height, width = img_data.shape

        # Validate dimensions to avoid index errors
        if depth < 3 or height < 3 or width < 3:
            raise ValueError(f"Image data is too small to slice: {img_data.shape}")

        slices = []
        if 'sagittal' in self.views:
            sagittal_slice = img_data[:, :, width // 2]
            slices.append(sagittal_slice)
        if 'coronal' in self.views:
            coronal_slice = img_data[:, height // 2, :]
            slices.append(coronal_slice)
        if 'axial' in self.views:
            axial_slice = img_data[depth // 2, :, :]
            slices.append(axial_slice)

        # Stack each slice into a single 3-channel image
        combined_image = np.stack(slices, axis=0)
        return combined_image
    
    def slice_and_aggregate(self, img_data):
        """
        Slices the 3D image from all three directions and aggregates them into the specified views.
        """
        # Ensure the image data is 3D (mri data)
        if img_data.ndim != 3:
            raise ValueError(f"Expected 3D image data, but got {img_data.ndim}D data.")

        depth, height, width = img_data.shape

        # Validate dimensions to avoid index errors
        if depth < 3 or height < 3 or width < 3:
            raise ValueError(f"Image data is too small to slice: {img_data.shape}")

        combined_images = []

        deviation = self.center_deviation

        if 'axial' in self.views:
            axial_deviation = int(deviation * width)
            axial_slices = [img_data[:, :, width // 2 - axial_deviation], 
                            img_data[:, :, width // 2], 
                            img_data[:, :, width // 2 + axial_deviation]]
            axial_image = np.stack(axial_slices, axis=0)
            combined_images.append(axial_image)

        if 'coronal' in self.views:
            coronal_deviation = int(deviation * height)
            coronal_slices = [img_data[:, height // 2 - coronal_deviation, :], 
                              img_data[:, height // 2, :], 
                              img_data[:, height // 2 + coronal_deviation, :]]
            coronal_image = np.stack(coronal_slices, axis=0)
            combined_images.append(coronal_image)

        if 'sagittal' in self.views:
            sagital_deviation = int(deviation * depth)
            sagittal_slices = [img_data[depth // 2 - sagital_deviation, :, :], 
                               img_data[depth // 2, :, :], 
                               img_data[depth // 2 + sagital_deviation, :, :]]
            sagittal_image = np.stack(sagittal_slices, axis=0)
            combined_images.append(sagittal_image)

        # Combine the directional images into a single multichannel image
        combined = np.concatenate(combined_images, axis=0)
        return combined
