"""
Dataset loader for CT and MRI heart segmentation
Handles NIfTI files for 4-chamber heart segmentation
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy import ndimage
import glob


class HeartSegmentationDataset(Dataset):
    """
    Dataset for loading CT and MRI heart images with segmentation labels
    Supports both modalities and handles 4-chamber segmentation
    """
    
    def __init__(self, data_dir, modality='ct', mode='train', transform=None, target_shape=(128, 128, 64)):
        """
        Args:
            data_dir: Root directory containing ct_train, ct_test, mr_train, mr_test folders
            modality: 'ct' or 'mr' for CT or MRI data
            mode: 'train' or 'test'
            transform: Optional transforms to apply
            target_shape: Target shape for resizing volumes (D, H, W)
        """
        self.data_dir = data_dir
        self.modality = modality.lower()
        self.mode = mode
        self.transform = transform
        self.target_shape = target_shape
        
        # Construct folder path
        folder_name = f"{self.modality}_{self.mode}"
        self.folder_path = os.path.join(data_dir, folder_name)
        
        # Get list of image files
        all_image_files = sorted([
            f for f in os.listdir(self.folder_path) 
            if f.endswith('_image.nii') and os.path.isfile(os.path.join(self.folder_path, f))
        ])
        
        # CRITICAL FIX: Filter out images without corresponding labels in train mode
        if self.mode == 'train':
            self.image_files = []
            missing_labels = []
            
            for img_file in all_image_files:
                label_file = img_file.replace('_image.nii', '_label.nii')
                label_path = os.path.join(self.folder_path, label_file)
                
                if os.path.exists(label_path):
                    self.image_files.append(img_file)
                else:
                    missing_labels.append(img_file)
            
            print(f"Found {len(self.image_files)} {modality.upper()} {mode} images with labels in {self.folder_path}")
            
            if missing_labels:
                print(f"⚠️  WARNING: {len(missing_labels)} images are MISSING labels:")
                for img in missing_labels[:5]:  # Show first 5
                    print(f"    - {img} (no corresponding label file)")
                if len(missing_labels) > 5:
                    print(f"    ... and {len(missing_labels)-5} more")
        else:
            self.image_files = all_image_files
            print(f"Found {len(self.image_files)} {modality.upper()} {mode} images in {self.folder_path}")
        
        # DIAGNOSTIC: Check what label values actually exist in the dataset
        if self.mode == 'train' and len(self.image_files) > 0:
            print(f"\n🔍 DIAGNOSING LABEL VALUES IN DATASET...")
            self._diagnose_labels()
            # Compute per-sample weights for sampling (oversample rare classes)
            self.sample_weights = self._compute_sample_weights()
        else:
            # Uniform weights for test or empty
            self.sample_weights = [1.0 for _ in range(len(self.image_files))]
    
    def _diagnose_labels(self):
        """
        Check what unique values exist in the first few label files
        This helps detect if label remapping is correct
        """
        print(f"Checking first 3 label files to detect label values...")
        all_unique_values = set()
        
        for i in range(min(3, len(self.image_files))):
            image_filename = self.image_files[i]
            label_filename = image_filename.replace('_image.nii', '_label.nii')
            label_path = os.path.join(self.folder_path, label_filename)
            
            if os.path.exists(label_path):
                label = self.load_nifti(label_path)
                unique_values = np.unique(label)
                all_unique_values.update(unique_values)
                print(f"  File {i+1}: Unique values = {sorted(unique_values)}")
        
        print(f"\n📊 ALL UNIQUE LABEL VALUES FOUND: {sorted(all_unique_values)}")
        
        # Check if expected values exist
        expected_values = {0, 205, 420, 500, 550, 600, 820, 850}
        found_expected = expected_values.intersection(all_unique_values)
        missing_expected = expected_values - all_unique_values
        
        print(f"✓ Expected values FOUND: {sorted(found_expected)}")
        if missing_expected:
            print(f"⚠️  Expected values MISSING: {sorted(missing_expected)}")
            print(f"💡 Your dataset might use different label encoding!")
            print(f"   Common alternatives:")
            print(f"   - Sequential: [0, 1, 2, 3, 4, 5, 6, 7]")
            print(f"   - Spaced: [0, 10, 20, 30, 40, 50, 60, 70]")
        print()

    def _compute_sample_weights(self):
        """Compute a sampling weight per sample to oversample volumes containing rare classes."""
        weights = []
        # Class bonuses (emphasize small/rare structures)
        bonuses = {
            1: 1.5,  # LV
            2: 1.5,  # RV
            5: 3.0,  # Myocardium
            7: 3.0   # PA
        }
        base = 1.0
        
        print("\n⚡ Computing sample weights (fast mode)...")
        
        for idx, img_file in enumerate(self.image_files):
            label_file = img_file.replace('_image.nii', '_label.nii')
            label_path = os.path.join(self.folder_path, label_file)
            w = base
            try:
                if os.path.exists(label_path):
                    # Fast mode: Load only a middle slice instead of full volume
                    nii = nib.load(label_path)
                    data = nii.get_fdata()
                    
                    # Sample middle slice only (much faster than full volume)
                    mid_slice = data.shape[0] // 2
                    label_slice = data[mid_slice, :, :]
                    label_slice = self.remap_labels(label_slice)
                    
                    uniques = set(np.unique(label_slice).tolist())
                    for c, b in bonuses.items():
                        if c in uniques:
                            w += b
            except Exception as e:
                # Fallback to base weight on error
                pass
            weights.append(float(w))
            
            # Progress indicator
            if (idx + 1) % 5 == 0 or (idx + 1) == len(self.image_files):
                print(f"  Processed {idx + 1}/{len(self.image_files)} samples...")
        
        # Normalize weights to mean=1 for stability
        if len(weights) > 0:
            w_arr = np.array(weights, dtype=np.float32)
            w_arr = w_arr / (w_arr.mean() + 1e-8)
            print(f"  ✓ Sample weights computed: min={w_arr.min():.2f}, max={w_arr.max():.2f}, mean={w_arr.mean():.2f}\n")
            return w_arr.tolist()
        return weights
    
    def __len__(self):
        return len(self.image_files)
    
    def load_nifti(self, filepath):
        """Load NIfTI file and return numpy array"""
        nii = nib.load(filepath)
        data = nii.get_fdata()
        return data
    
    def resize_volume(self, volume, target_shape, is_label=False):
        """Resize 3D volume to target shape using appropriate interpolation"""
        zoom_factors = [
            target_shape[i] / volume.shape[i] 
            for i in range(3)
        ]
        # Use nearest neighbor for labels to preserve class indices
        order = 0 if is_label else 1
        resized = ndimage.zoom(volume, zoom_factors, order=order)
        return resized
    
    def normalize_volume(self, volume):
        """Normalize volume to [0, 1] range"""
        volume = volume.astype(np.float32)
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
        return volume
    
    def __getitem__(self, idx):
        # Get image filename
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.folder_path, image_filename)
        
        # Load image
        image = self.load_nifti(image_path)
        
        # For training mode, load corresponding label
        if self.mode == 'train':
            label_filename = image_filename.replace('_image.nii', '_label.nii')
            label_path = os.path.join(self.folder_path, label_filename)
            
            if os.path.exists(label_path):
                label = self.load_nifti(label_path)
            else:
                print(f"Warning: Label not found for {image_filename}")
                label = np.zeros_like(image)
        else:
            label = np.zeros_like(image)  # Placeholder for test mode
        
        # Resize volumes
        image = self.resize_volume(image, self.target_shape, is_label=False)
        label = self.resize_volume(label, self.target_shape, is_label=True)
        
        # Normalize image
        image = self.normalize_volume(image)
        
        # CRITICAL FIX: Remap label values to [0, 1, 2, 3, 4, 5, 6, 7]
        label = self.remap_labels(label)
        
        # Ensure label is integer type and clip to valid range [0, 7]
        label = label.astype(np.int64)
        label = np.clip(label, 0, 7)
        
        # Add channel dimension for image (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # Simple data augmentation for training
        if self.mode == 'train' and np.random.rand() > 0.5:
            # Random horizontal flip
            image = torch.flip(image, dims=[3])
            label = torch.flip(label, dims=[2])
        
        if self.mode == 'train' and np.random.rand() > 0.5:
            # Random vertical flip
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return {
            'image': image,
            'label': label,
            'filename': image_filename
        }
    
    def remap_labels(self, label):
        """
        Remap label values from dataset-specific values to [0-7]
        
        UPDATED: Now checks if labels are already in [0-7] range
        """
        # Check if labels are already in correct format [0-7]
        unique_vals = np.unique(label)
        
        # If all values are already in range [0, 7], no remapping needed
        if np.all((unique_vals >= 0) & (unique_vals <= 7)):
            return label.astype(np.int64)
        
        # Original remapping for specific dataset encoding
        remapped = np.zeros_like(label, dtype=np.int64)
        
        # Map all 8 classes (background + 7 structures)
        remapped[label == 0] = 0      # Background
        remapped[label == 500] = 1    # Left Ventricle (LV) blood cavity
        remapped[label == 600] = 2    # Right Ventricle (RV) blood cavity
        remapped[label == 420] = 3    # Left Atrium (LA blood cavity)
        remapped[label == 550] = 4    # Right Atrium (RA) blood cavity
        remapped[label == 205] = 5    # Myocardium of the Left Ventricle
        remapped[label == 820] = 6    # Ascending Aorta (AO)
        remapped[label == 850] = 7    # Pulmonary Artery (PA)
        
        # Check if remapping worked
        remapped_unique = np.unique(remapped)
        if len(remapped_unique) <= 2:  # Only background + 1 class or less
            print(f"⚠️  WARNING: Remapping failed! Only found {remapped_unique}")
            print(f"   Original unique values were: {unique_vals}")
            print(f"   This means the label encoding doesn't match expectations!")
            
            # Try alternative: assume labels are sequential but offset
            if np.max(unique_vals) <= 10:
                print(f"   🔄 Attempting alternative: treating labels as sequential...")
                return label.astype(np.int64)
        
        return remapped


def get_dataloaders(data_dir, batch_size=2, num_workers=2, target_shape=(160, 160, 80)):  # Increased resolution
    """
    Create train and validation dataloaders for both CT and MR data
    
    Args:
        data_dir: Root directory containing the archive folder
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        target_shape: Target shape for volumes
    
    Returns:
        Dictionary with train and val dataloaders for CT and MR
    """
    from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
    
    # Create datasets for CT and MR
    ct_train_dataset = HeartSegmentationDataset(
        data_dir, modality='ct', mode='train', target_shape=target_shape
    )
    mr_train_dataset = HeartSegmentationDataset(
        data_dir, modality='mr', mode='train', target_shape=target_shape
    )
    
    # Combine CT and MR datasets for multi-modal training
    combined_train_dataset = ConcatDataset([ct_train_dataset, mr_train_dataset])
    
    # Build WeightedRandomSampler to oversample volumes containing rare classes
    combined_weights = []
    if hasattr(ct_train_dataset, 'sample_weights'):
        combined_weights.extend(ct_train_dataset.sample_weights)
    else:
        combined_weights.extend([1.0] * len(ct_train_dataset))
    if hasattr(mr_train_dataset, 'sample_weights'):
        combined_weights.extend(mr_train_dataset.sample_weights)
    else:
        combined_weights.extend([1.0] * len(mr_train_dataset))

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(combined_weights, dtype=torch.double),
        num_samples=len(combined_weights),
        replacement=True
    )

    # Create training dataloader with sampler (disable shuffle when using sampler)
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test datasets (optional for validation during training)
    ct_test_dataset = HeartSegmentationDataset(
        data_dir, modality='ct', mode='test', target_shape=target_shape
    )
    
    test_loader = DataLoader(
        ct_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'test': test_loader,
        'train_dataset': combined_train_dataset
    }


if __name__ == "__main__":
    # Test the dataset with CORRECT target shape
    data_dir = "./archive"
    
    # Use same target shape as training!
    dataset = HeartSegmentationDataset(data_dir, modality='ct', mode='train', target_shape=(160, 160, 80))
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Unique labels: {torch.unique(sample['label'])}")
    
    # CHECK: Count voxels per class
    label_np = sample['label'].numpy()
    print(f"\nVoxel count per class:")
    for class_id in range(8):
        count = np.sum(label_np == class_id)
        percentage = (count / label_np.size) * 100
        structure_names = {
            0: "Background", 1: "LV", 2: "RV", 3: "LA", 
            4: "RA", 5: "Myocardium", 6: "Aorta", 7: "PA"
        }
        print(f"  Class {class_id} ({structure_names[class_id]}): {count} voxels ({percentage:.2f}%)")
    
    print(f"Filename: {sample['filename']}")
