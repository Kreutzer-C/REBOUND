import numpy as np
import random
import torch
from scipy.ndimage import zoom, rotate
from monai.transforms import RandGaussianNoise, RandShiftIntensity
import albumentations


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
        has_next = 'next_image' in sample
        has_prev = 'prev_image' in sample

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
            if has_next:
                sample['next_image'] = zoom(sample['next_image'], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            if has_prev:
                sample['prev_image'] = zoom(sample['prev_image'], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            sample['mask'] = zoom(sample['mask'], (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(sample['mask'].astype(np.float32))
        sample = {'image': image, 'mask': label.long()}

        if has_next:
            sample['next_image'] = torch.from_numpy(sample['next_image'].astype(np.float32)).unsqueeze(0)
        if has_prev:
            sample['prev_image'] = torch.from_numpy(sample['prev_image'].astype(np.float32)).unsqueeze(0)

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
            # Keep numpy.ndarray format for val / test
            result = {
                'image': augmented['image'].astype(np.float32),
                'mask':  augmented['mask'].astype(np.float32),
            }
            if has_next:
                result['next_image'] = augmented['next_image'].astype(np.float32)
            if has_prev:
                result['prev_image'] = augmented['prev_image'].astype(np.float32)

        return result