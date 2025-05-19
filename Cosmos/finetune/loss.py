import torch
import pytorch_msssim

class SSIM3D(torch.nn.Module):
    def __init__(self):
        super(SSIM3D, self).__init__()

    def forward(self, img1, img2):
        # Clamp để đảm bảo giá trị nằm trong [0, 1]
        img1 = img1.to(torch.bfloat16).clamp(0, 1)
        img2 = img2.to(torch.bfloat16).clamp(0, 1)

        ssim_values = []

        for i in range(img1.shape[2]):  # D slice
            slice_img1 = img1[:, :, i, :, :]
            slice_img2 = img2[:, :, i, :, :]
            ssim_val = pytorch_msssim.ssim(slice_img1, slice_img2, data_range=1.0, size_average=True)
            ssim_values.append(ssim_val)

        return torch.mean(torch.stack(ssim_values))

