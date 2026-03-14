import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


def min_max_normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


class TripleSliceDataset(Dataset):
    """
    Dataset handler for 2.5D Triple Slice Dataset, designed to manage image and mask data for training and testing phases.

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

        image = min_max_normalize(image)
        next_slice = min_max_normalize(next_slice)
        prev_slice = min_max_normalize(prev_slice)

        sample = {'image': image, 'mask': mask, 'next_image': next_slice, 'prev_image': prev_slice}

        if self.transform:
            sample = self.transform(sample) # Apply transformations if specified
        sample['case_name'] = case_name
        sample['slice_name'] = slice_name

        return sample


class VolumeDataset(Dataset):
    """
    Dataset handler for 3D Volume Dataset, designed to manage image and mask data for training and testing phases.

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
