from ctvit import CTViT
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_new import MedicalImageReportDataset
import numpy as np
import torch.nn.functional as F

image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

mapping_label = {
    "abdomen_pelvis": 0,
    "chest": 1,
    "head_neck": 2,
}
image_encoder.load_state_dict(torch.load('/home/user01/aiotlab/htien/pet-clip/ViT_ckpts/CTVit.39000.pt'))
def process_image(image: np.ndarray, fix_depth=140):
    """
    Process the image from D x H x W to C x H x W x D
    - Resize the depth dimension to fix_depth using interpolation
    - Ensure fix_depth is divisible by 4 (pad if necessary)
    - Normalize pixel values by dividing by 32767
    - Convert image to (1, H, W, D) format
    
    Args:
        image (np.ndarray): The image with shape (D, H, W)
        fix_depth (int): The desired depth size
    
    Returns:
        torch.Tensor: Processed image with shape (1, H, W, D)
    """
    D, H, W = image.shape

    image_tensor = torch.tensor(image, dtype=torch.float32) / 32767.0

    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    image_tensor = F.interpolate(image_tensor, size=(fix_depth, 480, 480), mode='trilinear', align_corners=False)

    image_tensor = image_tensor.squeeze(0).squeeze(0)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

class CustomMedicalImageReportDataset(MedicalImageReportDataset):
    def __getitem__(self, idx):
        img_path, report_path = self.samples[idx]
        image = np.load(img_path)
        if self.transform:
            image = self.transform(image)
        else:
            image = process_image(image)
        # print(report_path)
        label = mapping_label[report_path.split('/')[-2]]
        
        return image, label
        
class Evaluate(nn.Module):
    def __init__(self, image_encoder, n_class):
        super().__init__()
        self.image_encoder = image_encoder
        self.n_class = n_class
        self.classifier = nn.Linear(294912, self.n_class)
    
    def forward(self, x):
        x = self.image_encoder(x)
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
def config(model):
    for param in model.parameters():
        param.requires_grad = False
    model.classifier.requires_grad = True
    return model

def train(model, data_loader, criterion, optimizer, device):
    model = config(model)
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Evaluate(image_encoder, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dataset = CustomMedicalImageReportDataset(root="/home/user01/aiotlab/thaind/DAC001_CTAC3.75mm_H_1001_PETWB3DAC001", split="train", transform=None)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for epoch in range(10):
        loss = train(model, data_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{10} - Loss: {loss:.4f}")