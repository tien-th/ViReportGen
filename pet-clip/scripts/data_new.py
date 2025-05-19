import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.ndimage import rotate
import random

import json
def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data



def add_gaussian_noise(image: np.array, mean=0, std=3276):
    """
    Add Gaussian noise to the image with given mean and standard deviation.
    
    Args:
        image (np.array): The image to add noise to
        mean (float): The mean of the Gaussian noise
        std (float): The standard deviation of the Gaussian noise
    
    Returns:
        np.array: The image with added Gaussian noise
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise

    return noisy_image


def augment_rotation(image: np.array): 
    rotation_axis = random.choice([0,1,2])
    rotation_angle = random.choice(range(-15, 16))

    np_img = rotate(image, rotation_angle, axes=(rotation_axis, (rotation_axis + 1) % 3), reshape=False)

    return np_img


def load_with_augment(image_path: str, augment: callable = None):
    image = np.load(image_path)

    if random.random() < 0.5:
        return image 
    else:
        organ = image_path.split('/')[-2]
        num_of_remove_slices = random.choice(range(10,21))
        if organ == 'chest':
            num_of_remove_slices_2 = random.choice(range(1,21))
            image = image[num_of_remove_slices:]
            image = image[:-num_of_remove_slices_2]
            # if image.shape[0] == 0: 
            #     with open('error.txt', 'a') as f:
            #         f.write(f"{num_of_remove_slices}_{num_of_remove_slices_2}_{image_path}\n")
            #     print(f"{num_of_remove_slices}_{num_of_remove_slices_2}_{image_path}\n")
        elif organ == 'abdomen_pelvis':
            image = image[num_of_remove_slices:]
        elif organ == 'head_neck':
            image = image[:-num_of_remove_slices]
        else:
            raise ValueError(f"Invalid organ: {organ}")
    
    if random.random() < 0.5 or augment is None:
        return image
    else:
        return augment(image)

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

    # Convert to torch tensor and normalize to [0, 1]
    image_tensor = torch.tensor(image, dtype=torch.float32) / 32767.0

    # Reshape to (1, 1, D, H, W) for interpolation (N, C, D, H, W)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    # Resize depth dimension using trilinear interpolation (D → fix_depth)
    image_tensor = F.interpolate(image_tensor, size=(fix_depth, 480, 480), mode='trilinear', align_corners=False)

    # Remove batch & channel dimensions → (fix_depth, H, W)
    image_tensor = image_tensor.squeeze(0)

    return image_tensor

def process_and_concat_2_images(img_path1, img_path2, fix_depth=140): 
    depth_per_img = fix_depth//2 
    img1 = np.load(img_path1)
    img2 = np.load(img_path2)
    img1_tensor = process_image(img1, fix_depth=depth_per_img)
    img2_tensor = process_image(img1, fix_depth=depth_per_img)

    concatenated_tensor = torch.cat((img1_tensor, img2_tensor), dim=1)

    return concatenated_tensor
    
    # concat through Depth dimension

    



from data_split import split_pet_data
class MedicalImageReportDataset(Dataset):
    def __init__(self, root, paraphase, comparation_path=None , augmentation=None, split='train'):
        """
        Args:
            root (str): Path to the root folder (e.g., "./DAC001").
            split (str): One of 'train', 'val', or 'test'.
                - train: use all month folders except THANG 10, THANG 11, THANG 12.
                - val: use only THANG 10.
                - test: use only THANG 11 and THANG 12.
            transform: Optional transform to be applied on a sample (e.g., conversion to torch tensor, normalization, etc.).
        """

        self.root = root
        self.paraphase = paraphase
        self.comparation_path = comparation_path
        self.comparation_length = 0
        self.augmentation = augmentation
        self.split = split.lower()
        
        
        # Determine which month folders to include based on the split.
        self.month_folders = []
        self.large_data_paths = []
        self.small_data_paths = []

        for year in os.listdir(root):
            year_path = os.path.join(root, year)
            if not os.path.isdir(year_path):
                continue
            if '2023' in year or '2018' in year:
                self.large_data_paths.append(year_path)
            elif '2019' in year or '2017' in year:
                self.small_data_paths.append(year_path)
            else:
                print(f"Unknown year folder: {year_path}")
                continue
        
        if comparation_path is not None: 
            self.comparation_data = read_jsonl_file(comparation_path)
            self.comparation_length = len(self.comparation_data)

        self.month_folders = split_pet_data(self.small_data_paths, self.large_data_paths, split=self.split)

        # Allowed modalities (exclude "whole_body")
        allowed_modalities = ['abdomen_pelvis', 'chest', 'head_neck']
        
        # Build the list of (image_path, report_path) pairs.
        self.samples = []
        for month_folder in self.month_folders:
            images_root = os.path.join(month_folder, 'images')
            # reports_root = os.path.join(month_folder, 'reports')
            # if not os.path.isdir(images_root) or not os.path.isdir(reports_root):
            #     continue
            for modality in allowed_modalities:
                modality_img_folder = os.path.join(images_root, modality)
                # modality_rep_folder = os.path.join(reports_root, modality)
                # if not os.path.isdir(modality_img_folder) or not os.path.isdir(modality_rep_folder):
                #     continue
                if not os.path.isdir(modality_img_folder):
                    continue
                # List all image files ending with .npy
                image_files = sorted([f for f in os.listdir(modality_img_folder) if f.endswith('.npy')])
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    rep_file = base_name + '.txt'
                    img_file_path = os.path.join(modality_img_folder, img_file)

                    if os.path.exists(img_file_path):
                        self.samples.append(img_file_path)
                    # rep_file_path = os.path.join(modality_rep_folder, rep_file)
                    # if os.path.exists(rep_file_path):
                    #     self.samples.append((img_file_path, rep_file_path))
                    #     if self.paraphase: 
                    #         paraphrase_report_path = rep_file_path.replace('PET_report_paired_fixed', 'paraphrase')
                    #         if os.path.exists(paraphrase_report_path):
                    #             self.samples.append((img_file_path, paraphrase_report_path))
                    # else:
                    #     print(f"paraphrase report not found: {paraphrase_report_path}")
        
        self.samples_length = len(self.samples)
        print('Original + paraphase data:', self.samples_length)
        print('Comparation data:', self.comparation_length)
        
        
    def __len__(self):
        return self.samples_length + self.comparation_length 
    
    def __getitem__(self, idx):
        if idx >= self.samples_length: 
            new_idx = idx - self.samples_length
            img_path1, img_path2 = self.comparation_data[new_idx]['file1'], self.comparation_data[new_idx]['file2']
            image = process_and_concat_2_images(img_path1, img_path2)
            # report = self.comparation_data[new_idx]['conclusion']
            return image
        else:  
            # img_path, report_path = self.samples[idx]
            img_path = self.samples[idx]
            # Load the image data from a .npy file
            # Apply augmentation if specified
            if self.augmentation:
                image = load_with_augment(img_path, augment=self.augmentation)
            else:
                image = np.load(img_path)
                
            image = process_image(image)
            # Load the report text.
            # with open(report_path, 'r', encoding='utf-8') as f:
            #     report = f.read().strip()
            # return image, report
            return image
