"""
Dataset & DataLoader for CAN+CBAM unified crowd density estimation.

KEY DIFFERENCE from CSRNet system:
    - MergedCrowdDataset combines Part A + Part B into ONE training set
    - WeightedRandomSampler ensures balanced sampling (50/50) from each part
    - Test sets remain SEPARATE for individual evaluation
    - Negative samples supported to reduce false positives

Enhanced with:
    - Segmentation-based preprocessing (optional)
    - Negative samples support (images with no people)
    - Robust data augmentation (scaling, flipping, cropping, color jitter)
    - Proper .mat annotation loading for density map generation
"""
import os
import glob
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config


class CrowdDataset(Dataset):
    """
    PyTorch Dataset for ShanghaiTech crowd counting (single part).

    Loads:
        - Image (JPG) from raw dataset
        - Density map (H5) from preprocessed directory

    Modes:
        - 'train': Random crop (256×256) + augmentation + optional negative samples
        - 'val'/'test': Full image (no crop, no augmentation)
    """

    def __init__(self, image_dir: str, h5_dir: str, mode: str = "train",
                 crop_size: int = 256, part: str = "A",
                 negative_samples_dir: str = None):
        """
        Args:
            image_dir:            Path to raw images/ folder
            h5_dir:               Path to preprocessed h5 ground-truth/ folder
            mode:                 'train' or 'test'
            crop_size:            Crop size for training (default 256)
            part:                 'A' or 'B' — used for tracking source dataset
            negative_samples_dir: Optional dir with images containing NO people
        """
        super().__init__()
        self.mode = mode
        self.crop_size = crop_size
        self.downsample = config.DOWNSAMPLE
        self.part = part.upper()

        # Find all image-h5 pairs
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.h5_paths = []
        self.gt_counts = []
        valid_pairs = []

        for img_p in self.image_paths:
            img_name = os.path.basename(img_p).replace(".jpg", "")
            
            # Try multiple possible h5 locations
            possible_h5_paths = [
                os.path.join(h5_dir, "density_maps", f"{img_name}.h5"),
                os.path.join(h5_dir, "ground-truth", f"{img_name}.h5"),
                os.path.join(h5_dir, "images", f"{img_name}.h5"),
                os.path.join(h5_dir, f"{img_name}.h5"),
            ]
            
            h5_p = None
            for p in possible_h5_paths:
                if os.path.exists(p):
                    h5_p = p
                    break
            
            if h5_p:
                # Read GT count for sampling weights
                gt_count = 0
                try:
                    with h5py.File(h5_p, 'r') as hf:
                        gt_count = hf.attrs.get('gt_count', 0)
                except Exception:
                    pass
                valid_pairs.append((img_p, h5_p, gt_count))
            else:
                # Optional: print(f"  ⚠ Missing h5 for {os.path.basename(img_p)}")
                pass

        self.image_paths = [p[0] for p in valid_pairs]
        self.h5_paths = [p[1] for p in valid_pairs]
        self.gt_counts = [p[2] for p in valid_pairs]

        # Track which part each sample belongs to
        self.part_labels = [self.part] * len(self.image_paths)

        # Dense/sparse labels for auxiliary classifier
        self.density_labels = [
            1.0 if c > config.DENSE_THRESHOLD else 0.0
            for c in self.gt_counts
        ]

        # ── Negative Samples (images with no people) ──
        self.negative_paths = []
        if negative_samples_dir and os.path.isdir(negative_samples_dir):
            self.negative_paths = sorted(
                glob.glob(os.path.join(negative_samples_dir, "*.jpg")) +
                glob.glob(os.path.join(negative_samples_dir, "*.png"))
            )
            if self.negative_paths:
                print(f"  ✓ Loaded {len(self.negative_paths)} negative samples")

        # Transforms
        self.normalize = T.Normalize(mean=config.IMAGENET_MEAN,
                                     std=config.IMAGENET_STD)

    def __len__(self):
        return len(self.image_paths) + len(self.negative_paths)

    def __getitem__(self, idx):
        # Check if this is a negative sample
        if idx >= len(self.image_paths):
            img, density, count, is_dense = self._get_negative_sample(
                idx - len(self.image_paths))
            return img, density, count, is_dense

        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # Load density map
        with h5py.File(self.h5_paths[idx], 'r') as hf:
            density = np.array(hf['density'], dtype=np.float32)
            gt_count = hf.attrs.get('gt_count', density.sum())

        # Dense/sparse label
        is_dense = torch.tensor(self.density_labels[idx], dtype=torch.float32)

        if self.mode == 'train':
            img_t, den_t, cnt_t = self._train_transform(img, density, gt_count)
            return img_t, den_t, cnt_t, is_dense
        else:
            img_t, den_t, cnt_t = self._test_transform(img, density, gt_count)
            return img_t, den_t, cnt_t, is_dense

    def _get_negative_sample(self, neg_idx):
        """
        Return a negative sample (no people) with zero density map.
        These help the model learn NOT to detect people in backgrounds.
        """
        img = Image.open(self.negative_paths[neg_idx]).convert('RGB')
        ds = self.downsample
        crop_size = self.crop_size

        if self.mode == 'train':
            img_w, img_h = img.size
            if img_w < crop_size or img_h < crop_size:
                scale = max(crop_size / img_w, crop_size / img_h) * 1.1
                img = img.resize(
                    (int(img_w * scale), int(img_h * scale)), Image.BILINEAR
                )

            img_w, img_h = img.size
            new_w = (img_w // ds) * ds
            new_h = (img_h // ds) * ds
            img = img.resize((new_w, new_h), Image.BILINEAR)

            crop_ds = crop_size // ds
            top = random.randint(0, new_h - crop_size)
            left = random.randint(0, new_w - crop_size)
            img = TF.crop(img, top, left, crop_size, crop_size)

            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                jitter = T.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1)
                img = jitter(img)

            img_tensor = TF.to_tensor(img)
            img_tensor = self.normalize(img_tensor)
            density_tensor = torch.zeros(1, crop_ds, crop_ds)
            count = torch.tensor(0.0, dtype=torch.float32)
        else:
            img_w, img_h = img.size
            new_w = (img_w // ds) * ds
            new_h = (img_h // ds) * ds
            img = img.resize((new_w, new_h), Image.BILINEAR)

            img_tensor = TF.to_tensor(img)
            img_tensor = self.normalize(img_tensor)
            density_tensor = torch.zeros(1, new_h // ds, new_w // ds)
            count = torch.tensor(0.0, dtype=torch.float32)

        is_dense = torch.tensor(0.0, dtype=torch.float32)
        return img_tensor, density_tensor, count, is_dense

    def _train_transform(self, img, density, gt_count):
        """Training: random crop, flip, scale, color jitter."""
        ds = self.downsample
        crop_size = self.crop_size

        img_w, img_h = img.size

        # ── Random scale augmentation: 0.8× to 1.2× ──
        if random.random() > 0.5:
            scale_factor = random.uniform(0.8, 1.2)
            new_w = int(img_w * scale_factor)
            new_h = int(img_h * scale_factor)
            new_w = max(new_w, crop_size)
            new_h = max(new_h, crop_size)
            new_w = (new_w // ds) * ds
            new_h = (new_h // ds) * ds
            img = img.resize((new_w, new_h), Image.BILINEAR)
            density = self._resize_density(density, new_h // ds, new_w // ds)
            img_w, img_h = new_w, new_h

        # Ensure image is large enough for cropping
        if img_w < crop_size or img_h < crop_size:
            scale = max(crop_size / img_w, crop_size / img_h) * 1.1
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            new_w = (new_w // ds) * ds
            new_h = (new_h // ds) * ds
            img = img.resize((new_w, new_h), Image.BILINEAR)
            density = self._resize_density(density, new_h // ds, new_w // ds)
            img_w, img_h = new_w, new_h

        # Make dimensions divisible by downsample
        new_w = (img_w // ds) * ds
        new_h = (img_h // ds) * ds
        if new_w != img_w or new_h != img_h:
            img = img.resize((new_w, new_h), Image.BILINEAR)
            density = self._resize_density(density, new_h // ds, new_w // ds)
            img_w, img_h = new_w, new_h

        # Random crop
        crop_ds = crop_size // ds
        density_h, density_w = density.shape

        if density_h <= crop_ds or density_w <= crop_ds:
            pad_h = max(0, crop_ds - density_h)
            pad_w = max(0, crop_ds - density_w)
            density = np.pad(density, ((0, pad_h), (0, pad_w)), mode='constant')
            img = TF.pad(img, (0, 0, pad_w * ds, pad_h * ds))
            density_h, density_w = density.shape

        top_d = random.randint(0, density_h - crop_ds)
        left_d = random.randint(0, density_w - crop_ds)
        top_i = top_d * ds
        left_i = left_d * ds

        img = TF.crop(img, top_i, left_i, crop_size, crop_size)
        density = density[top_d:top_d + crop_ds, left_d:left_d + crop_ds]

        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            density = np.fliplr(density).copy()

        # Color jitter
        if random.random() > 0.5:
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2,
                                    saturation=0.1)
            img = jitter(img)

        # Convert to tensors
        img_tensor = TF.to_tensor(img)
        img_tensor = self.normalize(img_tensor)
        density_tensor = torch.from_numpy(density).unsqueeze(0)
        count = torch.tensor(density.sum(), dtype=torch.float32)

        return img_tensor, density_tensor, count

    def _test_transform(self, img, density, gt_count):
        """Test/Val: full image, no augmentation."""
        ds = self.downsample
        img_w, img_h = img.size

        new_w = (img_w // ds) * ds
        new_h = (img_h // ds) * ds
        if new_w != img_w or new_h != img_h:
            img = img.resize((new_w, new_h), Image.BILINEAR)

        expected_h = new_h // ds
        expected_w = new_w // ds
        if density.shape != (expected_h, expected_w):
            density = self._resize_density(density, expected_h, expected_w)

        img_tensor = TF.to_tensor(img)
        img_tensor = self.normalize(img_tensor)
        density_tensor = torch.from_numpy(density).unsqueeze(0)
        count = torch.tensor(gt_count, dtype=torch.float32)

        return img_tensor, density_tensor, count

    @staticmethod
    def _resize_density(density, target_h, target_w):
        """Resize density map while preserving total count."""
        from PIL import Image as PILImage
        original_sum = density.sum()
        d_img = PILImage.fromarray(density)
        d_img = d_img.resize((target_w, target_h), PILImage.BILINEAR)
        density_resized = np.array(d_img, dtype=np.float32)
        new_sum = density_resized.sum()
        if new_sum > 0:
            density_resized *= (original_sum / new_sum)
        return density_resized


def collate_fn(batch):
    """
    Custom collate for variable-size test images.
    For training (fixed crop), default collate works.
    For test, we handle variable sizes by not stacking.
    """
    images, densities, counts, is_dense = zip(*batch)

    shapes = [img.shape for img in images]
    if len(set(shapes)) == 1:
        return (torch.stack(images), torch.stack(densities),
                torch.stack(counts), torch.stack(is_dense))
    else:
        return (list(images), list(densities),
                torch.stack(counts), torch.stack(is_dense))


def get_merged_train_loader(batch_size: int = None,
                            negative_samples_dir: str = None):
    """
    Create a MERGED training DataLoader with Density-Balanced Sampling.

    Balances between three categories to ensure high performance on sparse scenes:
        1. Negative (Empty) — from the negative samples directory
        2. Sparse (Part B) — low headcount (fixed sigma)
        3. Dense (Part A) — high headcount (adaptive sigma)

    Args:
        batch_size:            override config batch size
        negative_samples_dir:  path to negative sample images

    Returns:
        train_loader: DataLoader with balanced density sampling
        dataset_info: dict with counts per category
    """
    bs = batch_size or config.BATCH_SIZE
    neg_dir = negative_samples_dir or config.NEGATIVE_SAMPLES_DIR

    # 1. Load datasets
    raw_train_a, _, h5_train_a, _ = config.get_paths("A")
    train_a = CrowdDataset(
        image_dir=os.path.join(raw_train_a, "images"),
        h5_dir=h5_train_a,
        mode="train",
        crop_size=config.CROP_SIZE,
        part="A",
        negative_samples_dir=neg_dir if os.path.isdir(neg_dir) else None,
    )

    raw_train_b, _, h5_train_b, _ = config.get_paths("B")
    train_b = CrowdDataset(
        image_dir=os.path.join(raw_train_b, "images"),
        h5_dir=h5_train_b,
        mode="train",
        crop_size=config.CROP_SIZE,
        part="B",
        negative_samples_dir=None,  # Negatives only in A loader for Concat logic
    )

    # Calculate individual counts
    num_neg = len(train_a.negative_paths)
    num_a_real = len(train_a.image_paths)
    num_b_real = len(train_b.image_paths)
    
    # 2. Merge datasets
    merged_dataset = ConcatDataset([train_a, train_b])
    total_len = len(merged_dataset)

    # 3. Construct 3-Way Balanced Weights
    # We want equal probability for Negatives, Part A, and Part B
    # Target prob: 1/3 each
    weight_neg = (1.0/3.0) / num_neg if num_neg > 0 else 0
    weight_a = (1.0/3.0) / num_a_real if num_a_real > 0 else 0
    weight_b = (1.0/3.0) / num_b_real if num_b_real > 0 else 0

    # Fill weight array
    # train_a indices: [0..num_a_real-1] then [num_a_real..num_a_real+num_neg-1]
    # train_b indices: [num_a_total..num_a_total+num_b_real-1]
    weights = []
    
    # train_a weights (Real A then Negatives)
    weights.extend([weight_a] * num_a_real)
    weights.extend([weight_neg] * num_neg)
    
    # train_b weights (Real B)
    weights.extend([weight_b] * num_b_real)

    sample_weights = torch.tensor(weights, dtype=torch.float64)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        merged_dataset,
        batch_size=bs,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        collate_fn=collate_fn,
    )

    dataset_info = {
        'part_a_count': num_a_real,
        'part_b_count': num_b_real,
        'negatives': num_neg,
        'total_len': total_len
    }

    print(f"\n  Merged Training Dataset (Balanced):")
    print(f"    - Part A (Dense):   {num_a_real} images")
    print(f"    - Part B (Sparse):  {num_b_real} images")
    print(f"    - Negatives (Zero): {num_neg} images")
    print(f"    -> Sampling weights equalized to ~33% each")

    return train_loader, dataset_info


def get_test_loader(part: str):
    """
    Create a test DataLoader for a SINGLE part (evaluated separately).

    Args:
        part: 'A' or 'B'

    Returns:
        test_loader: DataLoader for the specified test set
    """
    _, raw_test, _, h5_test = config.get_paths(part)

    test_dataset = CrowdDataset(
        image_dir=os.path.join(raw_test, "images"),
        h5_dir=h5_test,
        mode="test",
        part=part,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    print(f"  Test DataLoader — Part {part.upper()}: "
          f"{len(test_dataset)} images, batch_size=1")

    return test_loader, test_dataset


if __name__ == "__main__":
    # Test merged training loader
    print("=" * 65)
    print("  Testing Merged Dataset")
    print("=" * 65)

    train_loader, info = get_merged_train_loader()

    print(f"\n  Testing merged train loader...")
    for imgs, densities, counts, is_dense in train_loader:
        if isinstance(imgs, torch.Tensor):
            print(f"    Batch — imgs: {imgs.shape}, density: {densities.shape}, "
                  f"counts: {counts}, dense: {is_dense}")
        else:
            print(f"    Batch — {len(imgs)} images (variable size), "
                  f"counts: {counts}")
        break

    # Test individual test loaders
    for part in ["A", "B"]:
        test_loader, test_dataset = get_test_loader(part)
        print(f"\n  Testing Part {part} test loader...")
        for imgs, densities, counts, is_dense in test_loader:
            if isinstance(imgs, torch.Tensor):
                print(f"    Sample — imgs: {imgs.shape}, "
                      f"density: {densities.shape}, count: {counts}")
            else:
                print(f"    Sample — variable size, count: {counts}")
            break

    print("\n  [DONE] All dataset tests passed!")
