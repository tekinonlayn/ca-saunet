import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class PlantDocSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = np.array(mask, dtype=np.uint8)
            mask = np.where(mask == 38, 1, 0)
            mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask

class ISICSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = torch.from_numpy((np.array(mask) > 128).astype('int64'))

        return image, mask

def get_data_loaders(dataset_name, data_dir, batch_size=16, seed=42):
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.where(np.array(x) == 38, 1, 0), dtype=torch.long) if dataset_name == 'plantdoc' else torch.from_numpy((np.array(x) > 128).astype('int64')))
    ])

    images_dir = os.path.join(data_dir, dataset_name, 'images')
    masks_dir = os.path.join(data_dir, dataset_name, 'masks')

    if dataset_name == 'plantdoc':
        dataset = PlantDocSegmentationDataset(images_dir, masks_dir, transform_img, transform_mask)
    elif dataset_name == 'isic':
        dataset = ISICSegmentationDataset(images_dir, masks_dir, transform_img, transform_mask)
    else:
        raise ValueError("Dataset must be 'plantdoc' or 'isic'")

    torch.manual_seed(seed)
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)