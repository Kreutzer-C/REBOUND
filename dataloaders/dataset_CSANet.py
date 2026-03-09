import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate
import SimpleITK as sitk
from monai.transforms import RandGaussianNoise, RandShiftIntensity
import albumentations


def min_max_normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


class CSANet_SliceDataset(Dataset):
    """
    Dataset handler for CSANet, designed to manage image and mask data for training and testing phases.

    Attributes:
        base_dir (str): Directory where image and mask data are stored.
        domain_name (str): Name of the domain to be used for training or testing.
        split (str): The current dataset split, indicating training or testing phase.
        metadata_path (str): Path to the metadata json file for current dataset.
        transform (callable, optional): A function/transform to apply to the samples.
    """
    def __init__(self, base_dir, domain_name, split, metadata, transform=None):
        self.base_dir = base_dir
        self.domain_name = domain_name
        self.split = split
        self.metadata = metadata
        self.transform = transform

        self.data_dir = os.path.join(self.base_dir, self.domain_name, 'slices')
        assert os.path.exists(self.data_dir), f"Data directory does not exist: {self.data_dir}"

        self.data_files = self._get_data_files()
        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {self.data_dir} for split '{self.split}'")
    
    def _get_data_files(self):
        """Get list of data files for the specified split."""
        files = []
        
        if self.metadata and "splits" in self.metadata:
            # Load specific case ids for the specified split
            case_ids = self.metadata["splits"][self.domain_name][self.split]
            for f in os.listdir(self.data_dir):
                if f.endswith('.npz'): 
                    # vol_0001_slice_0061.npz
                    case_id = f.split('_')[1]
                    if case_id in case_ids:
                        files.append(os.path.join(self.data_dir, f))
        else:
            # Load all data files if no splits are defined
            for f in os.listdir(self.data_dir):
                if f.endswith('.npz'):
                    files.append(os.path.join(self.data_dir, f))

        return sorted(files)
    
    def _get_next_and_prev_slices(self, case_name, slice_name):
        """Get the next and previous slices for a given case and slice name."""
        next_slice_name = str(int(slice_name) + 1).zfill(4)
        prev_slice_name = str(int(slice_name) - 1).zfill(4)
        next_slice_path = os.path.join(self.data_dir, f"vol_{case_name}_slice_{next_slice_name}.npz")
        prev_slice_path = os.path.join(self.data_dir, f"vol_{case_name}_slice_{prev_slice_name}.npz")

        if not os.path.exists(next_slice_path):
            next_slice_path = os.path.join(self.data_dir, f"vol_{case_name}_slice_{slice_name}.npz")
        if not os.path.exists(prev_slice_path):
            prev_slice_path = os.path.join(self.data_dir, f"vol_{case_name}_slice_{slice_name}.npz")

        return next_slice_path, prev_slice_path
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        case_name = data_file.split('/')[-1].split('_')[1]
        slice_name = data_file.split('/')[-1].split('_')[3].split('.')[0]

        data = np.load(data_file)
        image = data['img']
        mask = data['label']

        next_slice_path, prev_slice_path = self._get_next_and_prev_slices(case_name, slice_name)
        next_slice = np.load(next_slice_path)['img']
        prev_slice = np.load(prev_slice_path)['img']

        if isinstance(self.transform, RandomGenerator_new):
            image = min_max_normalize(image)
            next_slice = min_max_normalize(next_slice)
            prev_slice = min_max_normalize(prev_slice)

        sample = {'image': image, 'mask': mask, 'next_image': next_slice, 'prev_image': prev_slice}

        if self.transform:
            sample = self.transform(sample) # Apply transformations if specified
        sample['case_name'] = case_name
        sample['slice_name'] = slice_name

        return sample


class CSANet_VolumeDataset(Dataset):
    """
    Dataset handler for CSANet, designed to manage image and mask data for training and testing phases.

    Attributes:
        base_dir (str): Directory where image and mask data are stored.
        domain_name (str): Name of the domain to be used for training or testing.
        split (str): The current dataset split, indicating training or testing phase.
        metadata_path (str): Path to the metadata json file for current dataset.
        transform (callable, optional): A function/transform to apply to the samples.
    """
    def __init__(self, base_dir, domain_name, split, metadata, transform=None):
        self.base_dir = base_dir
        self.domain_name = domain_name
        self.split = split
        self.metadata = metadata
        self.transform = transform

        self.data_dir = os.path.join(self.base_dir, self.domain_name, 'volumes')
        assert os.path.exists(self.data_dir), f"Data directory does not exist: {self.data_dir}"

        self.data_files = self._get_data_files()
        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {self.data_dir} for split '{self.split}'")
    
    def _get_data_files(self):
        """Get list of data files for the specified split."""
        files = []
        
        if self.metadata and "splits" in self.metadata:
            # Load specific case ids for the specified split
            case_ids = self.metadata["splits"][self.domain_name][self.split]
            for f in os.listdir(self.data_dir):
                if f.startswith('img_') and f.endswith('.nii.gz'): 
                    # img_0001.nii.gz
                    case_id = f.split('_')[1].split('.')[0]
                    if case_id in case_ids:
                        files.append(os.path.join(self.data_dir, f))
        else:
            # Load all data files if no splits are defined
            for f in os.listdir(self.data_dir):
                if f.startswith('img_') and f.endswith('.nii.gz'):
                    files.append(os.path.join(self.data_dir, f))

        return sorted(files)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        case_name = data_file.split('/')[-1].split('_')[1].split('.')[0] # 0001

        image = sitk.ReadImage(data_file)
        image = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(data_file.replace('img_', 'label_'))
        mask = sitk.GetArrayFromImage(mask)

        image = min_max_normalize(image)

        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = case_name
        return sample


def spatial_augment(data, k=None, flip_axis=None, angle=None, is_mask=False):
    """
    Applies spatial transformations to the data.
    """
    if k is not None:
        data = np.rot90(data, k=k, axes=(0, 1))
    if flip_axis is not None:
        data = np.flip(data, axis=flip_axis)
    if angle is not None:
        data = rotate(data, angle, axes=(0, 1), reshape=False, order=0 if is_mask else 3)
    return data

def pixel_augment(data, noise_transform=None, brightness_transform=None):
    """
    Applies pixel-level transformations to the data (non-mask only) via MONAI transforms.

    Parameters:
        data (np.ndarray): Input image array.
        noise_transform (RandGaussianNoise, optional): MONAI transform for Gaussian noise.
        brightness_transform (RandShiftIntensity, optional): MONAI transform for brightness shift.
    """
    if noise_transform is not None:
        data = noise_transform(data)
    if brightness_transform is not None:
        data = brightness_transform(data)
    return data


class RandomGenerator(object):
    """
    Applies random transformations.

    Parameters:
        output_size (tuple): Desired output dimensions (height, width) for the images and labels.
    """
    def __init__(self, output_size, phase="train"):
        self.output_size = output_size
        self.phase = phase
        # MONAI pixel-level augmentation transforms (prob controls random apply internally)
        self.rand_gaussian_noise = RandGaussianNoise(prob=0.5, mean=0.0, std=0.1)
        self.rand_shift_intensity = RandShiftIntensity(offsets=0.2, prob=0.5)
    
    def __call__(self, sample):
        if self.phase == "train":
            # Apply spatial transformations to the data
            k = flip_axis = angle = None
            if random.random() > 0.5:
                k = np.random.randint(0, 4)
            if random.random() > 0.5:
                flip_axis = np.random.randint(0, 2)
            if random.random() > 0.5:
                angle = np.random.randint(-20, 20)
            
            for key in sample.keys():
                sample[key] = spatial_augment(sample[key], k, flip_axis, angle, key == 'mask')

            # Apply pixel-level augmentations to image channels only (not mask)
            for key in sample.keys():
                if key != 'mask':
                    sample[key] = pixel_augment(sample[key], self.rand_gaussian_noise, self.rand_shift_intensity)
            
        x, y = sample['image'].shape
        if x != self.output_size[0] or y != self.output_size[1]:
            print(f"Rescaling image from {x}x{y} to {self.output_size[0]}x{self.output_size[1]} (Not Recommended)")
            sample['image'] = zoom(sample['image'], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            sample['next_image'] = zoom(sample['next_image'], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            sample['prev_image'] = zoom(sample['prev_image'], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            sample['mask'] = zoom(sample['mask'], (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(sample['mask'].astype(np.float32))
        next_image = torch.from_numpy(sample['next_image'].astype(np.float32)).unsqueeze(0)
        prev_image = torch.from_numpy(sample['prev_image'].astype(np.float32)).unsqueeze(0)

        sample = {'image': image, 'mask': label.long(), 'next_image': next_image, 'prev_image': prev_image}
        return sample


class RandomGenerator_new(object):
    """
    Applies random transformations using albumentations library.

    Spatial augmentations (Resize, RandomResizedCrop, ShiftScaleRotate, ElasticTransform)
    are applied with **identical random parameters** to all 2.5D channels simultaneously
    (image, next_image, prev_image, mask), ensuring geometric consistency across slices.

    Pixel-level augmentations (RandomBrightnessContrast, RandomGamma, GaussNoise) are
    applied only to image-type channels (not mask), with the same random parameters shared
    across all image channels in each call.

    Parameters:
        output_size (tuple): Desired output dimensions (height, width).
        phase (str): 'train' applies full augmentation pipeline; 'val'/'test' resizes only.
    """
    def __init__(self, output_size, phase="train"):
        self.output_size = output_size
        self.phase = phase

        # Register next_image / prev_image as synchronized 'image'-type targets so that
        # ALL transforms (spatial + pixel) share the same random parameters across all channels.
        additional_targets = {
            'next_image': 'image',
            'prev_image': 'image',
        }

        if phase == "train":
            self.transform = albumentations.Compose([
                albumentations.Resize(output_size[0], output_size[1]),
                albumentations.RandomResizedCrop(
                    height=output_size[0], width=output_size[1],
                    scale=(0.9, 1.1), ratio=(0.9, 1.1), p=0.25
                ),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.05, rotate_limit=15, p=0.25
                ),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.25
                ),
                albumentations.RandomGamma(gamma_limit=(60, 140), p=0.25),
                albumentations.GaussNoise(p=0.25),
                albumentations.ElasticTransform(p=0.25),
            ], additional_targets=additional_targets)
        else:
            # val / test: resize only, no augmentation
            self.transform = albumentations.Compose([
                albumentations.Resize(output_size[0], output_size[1]),
            ], additional_targets=additional_targets)

    def __call__(self, sample):
        # Build albumentations input dict; handle optional 2.5D slice keys gracefully
        albu_input = {
            'image': sample['image'].astype(np.float32),
            'mask':  sample['mask'].astype(np.uint8),
        }
        has_next = 'next_image' in sample
        has_prev = 'prev_image' in sample
        if has_next:
            albu_input['next_image'] = sample['next_image'].astype(np.float32)
        if has_prev:
            albu_input['prev_image'] = sample['prev_image'].astype(np.float32)

        augmented = self.transform(**albu_input)

        if self.phase == 'train':
            # Convert to tensors; images get a channel dim (C=1), mask stays 2-D then cast to long
            result = {
                'image': torch.from_numpy(augmented['image'].astype(np.float32)).unsqueeze(0),
                'mask':  torch.from_numpy(augmented['mask'].astype(np.float32)).long(),
            }
            if has_next:
                result['next_image'] = torch.from_numpy(
                    augmented['next_image'].astype(np.float32)).unsqueeze(0)
            if has_prev:
                result['prev_image'] = torch.from_numpy(
                    augmented['prev_image'].astype(np.float32)).unsqueeze(0)
        else:
            result = {
                'image': augmented['image'].astype(np.float32),
                'mask':  augmented['mask'].astype(np.float32),
            }
            if has_next:
                result['next_image'] = augmented['next_image'].astype(np.float32)
            if has_prev:
                result['prev_image'] = augmented['prev_image'].astype(np.float32)

        return result
            

