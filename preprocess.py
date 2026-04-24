"""
Preprocessing — Generate Gaussian density maps from ShanghaiTech .mat annotations.

Ground Truth Construction:
    D(x) = Σᵢ N(x − xᵢ, σᵢ)

Part A (dense): Adaptive σ via k-nearest neighbor distances (fast binned approach)
Part B (sparse): Fixed σ = 15

OPTIMIZED: Uses sigma-binning to avoid per-point Gaussian filtering.
"""
import os
import glob
import h5py
import numpy as np
import scipy.io as sio
import scipy.ndimage
from scipy.spatial import KDTree
from PIL import Image
from tqdm import tqdm

import config


def load_ground_truth(mat_path: str) -> np.ndarray:
    """
    Load head annotations from a .mat file.
    Returns: (N, 2) array of [x, y] head positions.
    """
    mat = sio.loadmat(mat_path)
    gt = mat["image_info"][0, 0][0, 0][0]  # (N, 2) — [x, y] coordinates
    return gt.astype(np.float32)


def generate_density_map_adaptive_fast(img_shape, points, k=3, beta=0.3,
                                       num_bins=5):
    """
    FAST density map with adaptive Gaussian sigma (for dense crowds).

    Uses sigma-binning: groups points by similar sigma and applies one
    Gaussian filter per bin → ~50x faster than per-point approach.
    """
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    if len(points) == 1:
        x = min(max(int(round(points[0][0])), 0), w - 1)
        y = min(max(int(round(points[0][1])), 0), h - 1)
        density[y, x] = 1.0
        return scipy.ndimage.gaussian_filter(density, sigma=15.0,
                                              mode='constant')

    # Compute per-point sigma via KNN
    tree = KDTree(points.copy(), leafsize=2048)
    k_actual = min(k + 1, len(points))
    distances, _ = tree.query(points, k=k_actual)

    if k_actual > 1:
        sigmas = beta * np.mean(distances[:, 1:], axis=1)
        sigmas = np.maximum(sigmas, 1.0)
    else:
        sigmas = np.full(len(points), 15.0)

    sigma_min, sigma_max = sigmas.min(), sigmas.max()

    if sigma_max - sigma_min < 1.0:
        avg_sigma = sigmas.mean()
        for pt in points:
            x = min(max(int(round(pt[0])), 0), w - 1)
            y = min(max(int(round(pt[1])), 0), h - 1)
            density[y, x] += 1.0
        density = scipy.ndimage.gaussian_filter(density, sigma=avg_sigma,
                                                 mode='constant')
        return density

    bin_edges = np.linspace(sigma_min, sigma_max + 0.001, num_bins + 1)
    bin_indices = np.digitize(sigmas, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    for b in range(num_bins):
        mask = bin_indices == b
        if not mask.any():
            continue

        bin_points = points[mask]
        bin_sigma = sigmas[mask].mean()

        point_map = np.zeros((h, w), dtype=np.float32)
        for pt in bin_points:
            x = min(max(int(round(pt[0])), 0), w - 1)
            y = min(max(int(round(pt[1])), 0), h - 1)
            point_map[y, x] += 1.0

        point_map = scipy.ndimage.gaussian_filter(point_map, sigma=bin_sigma,
                                                   mode='constant',
                                                   truncate=4.0)
        density += point_map

    return density


def generate_density_map_fixed(img_shape, points, sigma=15.0):
    """
    Generate density map with fixed Gaussian sigma (for sparse crowds).
    """
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    for pt in points:
        x = min(max(int(round(pt[0])), 0), w - 1)
        y = min(max(int(round(pt[1])), 0), h - 1)
        density[y, x] += 1.0

    density = scipy.ndimage.gaussian_filter(density, sigma=sigma,
                                             mode='constant')
    return density


def preprocess_split(image_dir: str, gt_dir: str, h5_output_dir: str,
                     part: str, split_name: str):
    """Process all images in a split: generate density maps and save as .h5"""
    os.makedirs(os.path.join(h5_output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(h5_output_dir, "ground-truth"), exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if not image_paths:
        print(f"  ⚠ No images found in {image_dir}")
        return

    print(f"\n  Processing Part {part} — {split_name} ({len(image_paths)} images)")
    print(f"  Output: {h5_output_dir}")

    counts = []

    for img_path in tqdm(image_paths, desc=f"  Part {part}/{split_name}",
                         ncols=80):
        img = Image.open(img_path)
        img_w, img_h = img.size

        img_name = os.path.basename(img_path).replace(".jpg", "")
        gt_name = f"GT_{img_name}.mat"
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            print(f"    ⚠ GT not found: {gt_path}")
            continue

        points = load_ground_truth(gt_path)
        gt_count = len(points)
        counts.append(gt_count)

        if part.upper() == "A":
            density = generate_density_map_adaptive_fast(
                (img_h, img_w), points,
                k=config.SIGMA_K, beta=config.SIGMA_BETA
            )
        else:
            density = generate_density_map_fixed(
                (img_h, img_w), points,
                sigma=config.SIGMA_FIXED
            )

        # Downsample density map by 8x using sum-pooling
        ds = config.DOWNSAMPLE
        h_ds = img_h // ds
        w_ds = img_w // ds
        density_cropped = density[:h_ds * ds, :w_ds * ds]
        density_ds = density_cropped.reshape(h_ds, ds, w_ds, ds).sum(axis=(1, 3))

        h5_path = os.path.join(h5_output_dir, "ground-truth", f"{img_name}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('density', data=density_ds, compression='gzip')
            hf.attrs['image_path'] = img_path
            hf.attrs['gt_count'] = gt_count
            hf.attrs['density_sum'] = float(density_ds.sum())
            hf.attrs['original_h'] = img_h
            hf.attrs['original_w'] = img_w
            hf.attrs['part'] = part.upper()

    if counts:
        counts = np.array(counts)
        print(f"\n  ✓ Part {part}/{split_name} — Stats:")
        print(f"    Images:     {len(counts)}")
        print(f"    Total heads: {counts.sum():,}")
        print(f"    Mean count:  {counts.mean():.1f}")
        print(f"    Min count:   {counts.min()}")
        print(f"    Max count:   {counts.max()}")
        print(f"    Std count:   {counts.std():.1f}")


def preprocess_all():
    """Preprocess both Part A and Part B (train + test)."""
    print("=" * 65)
    print("  Density Map Preprocessing — ShanghaiTech Dataset")
    print("  (for CAN + CBAM)")
    print("=" * 65)

    for part in ["A", "B"]:
        raw_train, raw_test, h5_train, h5_test = config.get_paths(part)

        preprocess_split(
            image_dir=os.path.join(raw_train, "images"),
            gt_dir=os.path.join(raw_train, "ground-truth"),
            h5_output_dir=h5_train,
            part=part, split_name="train"
        )

        preprocess_split(
            image_dir=os.path.join(raw_test, "images"),
            gt_dir=os.path.join(raw_test, "ground-truth"),
            h5_output_dir=h5_test,
            part=part, split_name="test"
        )

    print("\n" + "=" * 65)
    print("  ✓ Preprocessing complete!")
    print("=" * 65)


if __name__ == "__main__":
    preprocess_all()
