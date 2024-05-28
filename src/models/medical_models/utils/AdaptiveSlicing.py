# import torch
# import torch.nn as nn

import torch
import torch.nn as nn

class AdaptiveSlicingModule(nn.Module):
    def __init__(self, initial_deviation=0.05):
        super(AdaptiveSlicingModule, self).__init__()
        self.deviation = nn.Parameter(torch.tensor(initial_deviation))

    def forward(self, x):
        # print("AdaptiveSlicingModule forward2: ", x.size())
        # N, C, H, W, D = x.size()  # Batch size, Channels, Height, Width, Depth
        # print("self.deviation: ", self.deviation.item())
        N, C, H, W, D = x.size()  # Batch size, Channels, Height, Width, Depth
        central_slice = D // 2
        offset = int(D * self.deviation.item() / 2)
        
        axial_slices = [max(0, central_slice - offset), central_slice, min(D - 1, central_slice + offset)]
        # print("Axial slices: ", axial_slices)
        axial_selected = torch.stack([x[:, :, :, :, i] for i in axial_slices], dim=1).squeeze(2)
        # print("Axial selected: ", axial_selected.shape)
        
        coronal_slices = [max(0, central_slice - offset), central_slice, min(H - 1, central_slice + offset)]
        coronal_selected = torch.stack([x[:, :, i, :, :] for i in coronal_slices], dim=1).squeeze(2)
        # print("Coronal selected: ", coronal_selected.shape)
        
        sagittal_slices = [max(0, central_slice - offset), central_slice, min(W - 1, central_slice + offset)]
        sagittal_selected = torch.stack([x[:, :, :, i, :] for i in sagittal_slices], dim=1).squeeze(2)
        # print("Sagittal selected: ", sagittal_selected.shape)
        
        # Concatenate all selected slices along the channel dimension
        combined_slices = torch.cat([axial_selected, coronal_selected, sagittal_selected], dim=1)
        # size should be torch.Size([N, 9, H, W])
        
        return combined_slices