"""
Evaluation metrics for medical image segmentation.
"""

import numpy as np
import torch
from scipy import ndimage


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute Dice coefficient.
    
    Args:
        pred: predicted segmentation (binary or probability)
        target: ground truth segmentation (binary)
        smooth: smoothing factor
    
    Returns:
        Dice coefficient value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def compute_dice_per_class(pred, target, num_classes, include_background=False):
    """
    Compute Dice coefficient for each class.
    
    Args:
        pred: (D, H, W) predicted segmentation (class indices)
        target: (D, H, W) ground truth segmentation (class indices)
        num_classes: number of classes
        include_background: whether to include background class (0)
    
    Returns:
        dict: {class_idx: dice_value}
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    dice_scores = {}
    start_class = 0 if include_background else 1
    
    for c in range(start_class, num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        if np.sum(target_c) == 0 and np.sum(pred_c) == 0:
            # Both empty, perfect match
            dice_scores[c] = 1.0
        elif np.sum(target_c) == 0 or np.sum(pred_c) == 0:
            # One is empty, the other is not
            dice_scores[c] = 0.0
        else:
            dice_scores[c] = dice_coefficient(pred_c, target_c)
    
    return dice_scores


def compute_surface_distances(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute surface distances between prediction and target.
    
    Args:
        pred: binary prediction mask
        target: binary target mask
        spacing: voxel spacing (z, y, x)
    
    Returns:
        tuple: (distances from pred to target, distances from target to pred)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Get surface voxels using morphological gradient
    pred_surface = pred ^ ndimage.binary_erosion(pred)
    target_surface = target ^ ndimage.binary_erosion(target)
    
    # Get coordinates of surface voxels
    pred_coords = np.array(np.where(pred_surface)).T * np.array(spacing)
    target_coords = np.array(np.where(target_surface)).T * np.array(spacing)
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return np.array([np.inf]), np.array([np.inf])
    
    # Compute distances from each pred surface point to nearest target surface point
    from scipy.spatial import cKDTree
    
    target_tree = cKDTree(target_coords)
    pred_to_target_distances, _ = target_tree.query(pred_coords)
    
    pred_tree = cKDTree(pred_coords)
    target_to_pred_distances, _ = pred_tree.query(target_coords)
    
    return pred_to_target_distances, target_to_pred_distances


def compute_assd(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Average Symmetric Surface Distance (ASSD).
    
    Args:
        pred: binary prediction mask
        target: binary target mask
        spacing: voxel spacing (z, y, x)
    
    Returns:
        ASSD value in mm
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Handle empty masks
    if np.sum(pred) == 0 and np.sum(target) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return np.inf
    
    pred_to_target, target_to_pred = compute_surface_distances(pred, target, spacing)
    
    if np.any(np.isinf(pred_to_target)) or np.any(np.isinf(target_to_pred)):
        return np.inf
    
    assd = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2.0
    return assd