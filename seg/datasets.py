import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

class CityscapesDataset(Dataset):
    """Cityscapes dataset for semantic segmentation"""
    
    # Cityscapes class definitions
    classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]
    
    # Class ID mapping
    id_to_trainid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
        24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
        31: 16, 32: 17, 33: 18
    }
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.targets = []
        
        # Build data paths
        img_dir = os.path.join(root, 'leftImg8bit', split)
        target_dir = os.path.join(root, 'gtFine', split)
        
        # Collect all image and label paths
        for city in os.listdir(img_dir):
            img_city_dir = os.path.join(img_dir, city)
            target_city_dir = os.path.join(target_dir, city)
            
            for fname in os.listdir(img_city_dir):
                if fname.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(img_city_dir, fname)
                    target_name = fname.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    target_path = os.path.join(target_city_dir, target_name)
                    
                    self.images.append(img_path)
                    self.targets.append(target_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        target_path = self.targets[idx]
        
        # Load image and label
        image = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)
        
        # Convert label IDs
        target = np.array(target)
        new_target = np.zeros_like(target)
        for k, v in self.id_to_trainid.items():
            new_target[target == k] = v
        target = Image.fromarray(new_target.astype(np.uint8))
        
        # Apply data augmentation
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target

class SegmentationTransform:
    """Data augmentation class for segmentation"""
    def __init__(self, split='train'):
        self.split = split
        self.target_size = (256, 512)  # height x width
        
        self.normalize = transforms.Normalize(
            mean=CityscapesDataset.mean,
            std=CityscapesDataset.std
        )
    
    def __call__(self, image, target):
        # Resize images
        image = F.resize(image, self.target_size, Image.BILINEAR)
        target = F.resize(target, self.target_size, Image.NEAREST)
        
        if self.split == 'train':
            # Random horizontal flip
            if np.random.random() < 0.5:
                image = F.hflip(image)
                target = F.hflip(target)
        
        # Convert to tensor
        image = F.to_tensor(image)
        target = torch.from_numpy(np.array(target)).long()
        
        # Normalize image
        image = self.normalize(image)
        
        return image, target 