"""
Utility Functions — Advanced visualization, metrics, peak detection,
circle-based head localization, Grad-CAM, and density map operations.

Provides shared utilities for training, testing, and the web application.
"""
import os
import io
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from PIL import Image
import torchvision.transforms as T

import config


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_counts, gt_counts):
    """
    Compute crowd counting evaluation metrics.

    Args:
        pred_counts: numpy array of predicted counts
        gt_counts:   numpy array of ground truth counts

    Returns:
        dict with MAE, MSE, RMSE, MAPE, R²
    """
    errors = pred_counts - gt_counts
    abs_errors = np.abs(errors)

    mae = abs_errors.mean()
    mse = (errors ** 2).mean()
    rmse = np.sqrt(mse)

    nonzero = gt_counts > 0
    if nonzero.sum() > 0:
        mape = (abs_errors[nonzero] / gt_counts[nonzero]).mean() * 100
    else:
        mape = 0.0

    ss_res = ((gt_counts - pred_counts) ** 2).sum()
    ss_tot = ((gt_counts - gt_counts.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def compute_ssim_psnr(pred_density, gt_density):
    """Compute SSIM and PSNR between density maps."""
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        pred_np = pred_density.squeeze().cpu().numpy()
        gt_np = gt_density.squeeze().cpu().numpy()

        if pred_np.shape != gt_np.shape:
            return 0.0, 0.0

        max_val = max(pred_np.max(), gt_np.max(), 1e-8)
        pred_norm = pred_np / max_val
        gt_norm = gt_np / max_val

        s = ssim(gt_norm, pred_norm, data_range=1.0)
        p = (psnr(gt_norm, pred_norm, data_range=1.0)
             if gt_norm.max() > 0 else 0.0)

        return float(s), float(p)
    except Exception:
        return 0.0, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ──────────────────────────────────────────────────────────────────────────────

def denormalize_image(tensor):
    """
    Reverse ImageNet normalization for display.
    Input:  (3, H, W) normalized tensor
    Output: (H, W, 3) numpy array in [0, 1]
    """
    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(config.IMAGENET_MEAN, config.IMAGENET_STD)],
        std=[1 / s for s in config.IMAGENET_STD]
    )
    img = inv_normalize(tensor.cpu().clone())
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def density_to_heatmap(density_np, cmap='jet'):
    """
    Convert density map to a colored heatmap image.
    Input:  (H, W) density map
    Output: (H, W, 3) RGB heatmap in [0, 1]
    """
    if density_np.max() > 0:
        density_norm = density_np / density_np.max()
    else:
        density_norm = density_np
    colormap = cm.get_cmap(cmap)
    heatmap = colormap(density_norm)[:, :, :3]
    return heatmap.astype(np.float32)


def create_overlay(image_np, density_np, alpha=0.5):
    """
    Overlay a density heatmap on an original image.

    Args:
        image_np:   (H, W, 3) image in [0, 1]
        density_np: (H_d, W_d) density map
        alpha:      blending factor

    Returns:
        overlay: (H, W, 3) blended image in [0, 1]
    """
    h, w = image_np.shape[:2]

    # Upsample density to match image size
    density_pil = Image.fromarray(density_np)
    density_up = np.array(density_pil.resize((w, h), Image.BILINEAR))

    heatmap = density_to_heatmap(density_up)

    overlay = (1 - alpha) * image_np + alpha * heatmap
    return np.clip(overlay, 0, 1)


def create_error_map(pred_density, gt_density):
    """
    Compute absolute difference error map between prediction and ground truth.
    """
    return np.abs(pred_density - gt_density)


# ──────────────────────────────────────────────────────────────────────────────
# Peak Detection — Convert Density Map → Head Positions
# ──────────────────────────────────────────────────────────────────────────────

def detect_peaks(density_np, min_distance=3, threshold_rel=0.1):
    """
    Detect local maxima in a density map to find individual head positions.

    Uses scipy's maximum_filter for non-maximum suppression.

    Args:
        density_np:    (H, W) density map (float)
        min_distance:  Minimum distance between detected peaks
        threshold_rel: Relative threshold (fraction of max value)

    Returns:
        peaks: (N, 2) array of [y, x] peak coordinates
    """
    from scipy.ndimage import maximum_filter, label

    if density_np.max() == 0:
        return np.array([]).reshape(0, 2)

    # Non-maximum suppression via maximum filter
    size = 2 * min_distance + 1
    local_max = maximum_filter(density_np, size=size)
    detected = (density_np == local_max)

    # Apply threshold
    abs_threshold = density_np.max() * threshold_rel
    detected = detected & (density_np > abs_threshold)

    # Extract peak coordinates
    coords = np.argwhere(detected)  # (N, 2) — [row, col] = [y, x]

    return coords


def get_adaptive_radius(density_np, peaks, base_radius=8,
                        dense_radius=3, sparse_radius=15):
    """
    Compute adaptive circle radius for each detected peak.

    Dense regions → small circles
    Sparse regions → larger circles

    Args:
        density_np:    (H, W) density map
        peaks:         (N, 2) array of [y, x] peak coordinates
        base_radius:   Default radius
        dense_radius:  Minimum radius for dense regions
        sparse_radius: Maximum radius for sparse regions

    Returns:
        radii: (N,) array of radii for each peak
    """
    if len(peaks) == 0:
        return np.array([])

    # Use local density to determine radius
    total_count = density_np.sum()
    h, w = density_np.shape

    if total_count == 0:
        return np.full(len(peaks), base_radius)

    # Compute local density around each peak
    local_densities = []
    window = 5
    for y, x in peaks:
        y0 = max(0, y - window)
        y1 = min(h, y + window + 1)
        x0 = max(0, x - window)
        x1 = min(w, x + window + 1)
        local_d = density_np[y0:y1, x0:x1].sum()
        local_densities.append(local_d)

    local_densities = np.array(local_densities)

    if local_densities.max() == 0:
        return np.full(len(peaks), base_radius)

    # Normalize and invert: high density → small radius
    norm_density = local_densities / (local_densities.max() + 1e-8)
    radii = sparse_radius - (sparse_radius - dense_radius) * norm_density
    radii = np.clip(radii, dense_radius, sparse_radius)

    return radii


# ──────────────────────────────────────────────────────────────────────────────
# Circle-Based Head Visualization
# ──────────────────────────────────────────────────────────────────────────────

def draw_head_circles(image_np, density_np, min_distance=3,
                      threshold_rel=0.1, upsample=True):
    """
    Draw adaptive-radius circles at detected head locations on the image.

    Args:
        image_np:      (H, W, 3) image in [0, 1]
        density_np:    (H_d, W_d) density map
        min_distance:  Minimum peak separation
        threshold_rel: Relative threshold for peak detection
        upsample:      Whether to upsample density to match image resolution

    Returns:
        fig: matplotlib figure with circles drawn
        num_detected: number of heads detected
    """
    h_img, w_img = image_np.shape[:2]

    # Upsample density to image resolution for accurate peak placement
    if upsample:
        density_up = np.array(
            Image.fromarray(density_np).resize((w_img, h_img), Image.BILINEAR)
        )
    else:
        density_up = density_np

    # Detect peaks
    peaks = detect_peaks(density_up, min_distance=min_distance,
                         threshold_rel=threshold_rel)
    radii = get_adaptive_radius(density_up, peaks)

    # Draw
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(image_np)

    for i, (y, x) in enumerate(peaks):
        r = radii[i] if i < len(radii) else 5
        circle = Circle((x, y), radius=r, fill=False,
                         edgecolor='lime', linewidth=1.5, alpha=0.85)
        ax.add_patch(circle)
        # Small dot at center
        ax.plot(x, y, 'o', color='red', markersize=1.5, alpha=0.7)

    ax.set_title(f'Detected Heads: {len(peaks)}', fontsize=14,
                 fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    return fig, len(peaks)


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive 7-Panel Visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_prediction_full(image_tensor, gt_density_tensor,
                               pred_density_tensor, gt_count, pred_count,
                               mask_np=None, save_path=None, title=None):
    """
    Generate comprehensive 7-panel visualization:
        1. Original image
        2. Segmented image (if mask provided)
        3. Ground truth density map
        4. Predicted density map
        5. Heatmap overlay on original image
        6. Circle-based head detection
        7. Error map (absolute difference)

    Also displays GT count, predicted count, and error.
    """
    img_np = denormalize_image(image_tensor)
    gt_np = gt_density_tensor.squeeze().cpu().numpy()
    pred_np = pred_density_tensor.squeeze().cpu().numpy()
    error = abs(pred_count - gt_count)

    # Determine number of panels
    has_mask = mask_np is not None
    n_panels = 7 if has_mask else 6

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    idx = 0

    # 1. Original image
    axes[idx].imshow(img_np)
    axes[idx].set_title('Original Image', fontsize=12)
    axes[idx].axis('off')
    idx += 1

    # 2. Segmented image (if mask available)
    if has_mask:
        axes[idx].imshow(mask_np, cmap='hot', vmin=0, vmax=1)
        axes[idx].set_title('Segmentation Mask', fontsize=12)
        axes[idx].axis('off')
        idx += 1

    # 3. Ground truth density
    axes[idx].imshow(gt_np, cmap='jet')
    axes[idx].set_title(f'GT Density\n(Count: {gt_count:.0f})', fontsize=12)
    axes[idx].axis('off')
    idx += 1

    # 4. Predicted density
    axes[idx].imshow(pred_np, cmap='jet')
    axes[idx].set_title(f'Pred Density\n(Count: {pred_count:.1f})', fontsize=12)
    axes[idx].axis('off')
    idx += 1

    # 5. Heatmap overlay
    overlay = create_overlay(img_np, pred_np, alpha=0.5)
    axes[idx].imshow(overlay)
    axes[idx].set_title('Heatmap Overlay', fontsize=12)
    axes[idx].axis('off')
    idx += 1

    # 6. Circle-based head detection
    h_img, w_img = img_np.shape[:2]
    density_up = np.array(
        Image.fromarray(pred_np).resize((w_img, h_img), Image.BILINEAR)
    )
    peaks = detect_peaks(density_up, min_distance=3, threshold_rel=0.1)
    radii = get_adaptive_radius(density_up, peaks)

    axes[idx].imshow(img_np)
    for i, (y, x) in enumerate(peaks):
        r = radii[i] if i < len(radii) else 5
        circle = Circle((x, y), radius=r, fill=False,
                         edgecolor='lime', linewidth=1.2, alpha=0.8)
        axes[idx].add_patch(circle)
    axes[idx].set_title(f'Head Detection\n({len(peaks)} heads)', fontsize=12)
    axes[idx].axis('off')
    idx += 1

    # 7. Error map
    error_map = create_error_map(pred_np, gt_np)
    axes[idx].imshow(error_map, cmap='hot')
    axes[idx].set_title(f'Error Map\n(AE: {error:.1f})', fontsize=12)
    axes[idx].axis('off')

    # Count info text box
    textstr = (f'GT: {gt_count:.0f}  |  Pred: {pred_count:.1f}  |  '
               f'Error: {error:.1f}  |  Peaks: {len(peaks)}')
    fig.text(0.5, -0.02, textstr, ha='center', fontsize=14,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig


def visualize_prediction(image_tensor, gt_density_tensor,
                          pred_density_tensor, gt_count, pred_count,
                          save_path=None, title=None):
    """
    Generate 5-panel visualization (backward compatible).
    """
    return visualize_prediction_full(
        image_tensor, gt_density_tensor, pred_density_tensor,
        gt_count, pred_count, mask_np=None,
        save_path=save_path, title=title
    )


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM Visualization
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CAN+CBAM.

    Visualizes which image regions contribute most to the density prediction.
    Hooks into the final CBAM output (before the 1×1 output conv).
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model:        CAN+CBAM model
            target_layer: layer to hook (default: model.cbam_backend)
        """
        self.model = model
        self.target_layer = target_layer or model.cbam_backend
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) input image tensor

        Returns:
            cam: (H, W) Grad-CAM heatmap normalized to [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            density_map = output[0]
        else:
            density_map = output

        # Use total density sum as target
        target = density_map.sum()

        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # Only positive contributions
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def visualize(self, input_tensor, image_np, save_path=None):
        """
        Generate and visualize Grad-CAM overlay.

        Args:
            input_tensor: (1, 3, H, W) input image tensor
            image_np:     (H, W, 3) original image in [0, 1]
            save_path:    path to save figure

        Returns:
            fig: matplotlib figure
        """
        cam = self.generate(input_tensor)

        h_img, w_img = image_np.shape[:2]
        cam_resized = np.array(
            Image.fromarray(cam).resize((w_img, h_img), Image.BILINEAR)
        )

        # Create overlay
        heatmap = cm.jet(cam_resized)[:, :, :3]
        overlay = 0.5 * image_np + 0.5 * heatmap

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(image_np)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
        axes[1].axis('off')

        axes[2].imshow(np.clip(overlay, 0, 1))
        axes[2].set_title('Grad-CAM Overlay', fontsize=14)
        axes[2].axis('off')

        plt.suptitle('CAN+CBAM — Grad-CAM Visualization', fontsize=16,
                     fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Training Curve Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, output_dir):
    """
    Plot comprehensive training curves.
    Generates: loss curves, MAE/RMSE for both Part A and Part B, and LR.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('CAN + CBAM — Unified Training Curves',
                 fontsize=18, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', lw=2, label='Train')
    if history.get('val_loss_A'):
        val_epochs = [e for e in epochs if (e % config.VALIDATE_EVERY == 0
                      or e == 1)]
        val_epochs = val_epochs[:len(history['val_loss_A'])]
        axes[0, 0].plot(val_epochs, history['val_loss_A'], 'r-o', lw=2,
                        label='Val A', markersize=4)
    if history.get('val_loss_B'):
        val_epochs = val_epochs[:len(history['val_loss_B'])]
        axes[0, 0].plot(val_epochs, history['val_loss_B'], 'g-s', lw=2,
                        label='Val B', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    if history.get('val_mae_A'):
        val_epochs_a = list(range(1, len(history['val_mae_A']) + 1))
        axes[0, 1].plot(val_epochs_a, history['val_mae_A'], 'r-o', lw=2,
                        label='Part A', markersize=4)
        best_a = min(history['val_mae_A'])
        axes[0, 1].axhline(y=best_a, color='r', linestyle='--', alpha=0.3)
    if history.get('val_mae_B'):
        val_epochs_b = list(range(1, len(history['val_mae_B']) + 1))
        axes[0, 1].plot(val_epochs_b, history['val_mae_B'], 'g-s', lw=2,
                        label='Part B', markersize=4)
        best_b = min(history['val_mae_B'])
        axes[0, 1].axhline(y=best_b, color='g', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Validation Check')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE
    if history.get('val_rmse_A'):
        axes[0, 2].plot(val_epochs_a, history['val_rmse_A'], 'r-o', lw=2,
                        label='Part A', markersize=4)
    if history.get('val_rmse_B'):
        axes[0, 2].plot(val_epochs_b, history['val_rmse_B'], 'g-s', lw=2,
                        label='Part B', markersize=4)
    axes[0, 2].set_xlabel('Validation Check')
    axes[0, 2].set_ylabel('RMSE')
    axes[0, 2].set_title('Root Mean Squared Error')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # R²
    if history.get('val_r2_A'):
        axes[1, 0].plot(val_epochs_a, history['val_r2_A'], 'r-o', lw=2,
                        label='Part A', markersize=4)
    if history.get('val_r2_B'):
        axes[1, 0].plot(val_epochs_b, history['val_r2_B'], 'g-s', lw=2,
                        label='Part B', markersize=4)
    axes[1, 0].set_xlabel('Validation Check')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('R² Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Aux classifier accuracy (if available)
    if history.get('aux_accuracy'):
        aux_epochs = list(range(1, len(history['aux_accuracy']) + 1))
        axes[1, 1].plot(aux_epochs, history['aux_accuracy'], 'purple', lw=2)
        axes[1, 1].set_xlabel('Validation Check')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Aux Dense/Sparse Classifier')
        axes[1, 1].set_ylim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, 'No aux classifier data',
                        ha='center', va='center', fontsize=14,
                        color='gray')
    axes[1, 1].grid(True, alpha=0.3)

    # LR
    axes[1, 2].plot(epochs, history['lr'], 'k-', lw=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Training curves saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Inference Helpers
# ──────────────────────────────────────────────────────────────────────────────

def prepare_image_for_inference(pil_image, device=None):
    """
    Prepare a PIL image for model inference.
    Makes dimensions divisible by 8 and normalizes.

    Args:
        pil_image: PIL Image (RGB)
        device:    torch device

    Returns:
        tensor: (1, 3, H, W) normalized image tensor
        original_size: (W, H) original dimensions
    """
    device = device or config.DEVICE
    ds = config.DOWNSAMPLE

    img = pil_image.convert('RGB')
    w, h = img.size
    original_size = (w, h)

    new_w = (w // ds) * ds
    new_h = (h // ds) * ds
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BILINEAR)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, original_size


def density_map_to_base64(density_np, cmap='jet'):
    """Convert a density map to a base64-encoded PNG image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(density_np, cmap=cmap)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


def overlay_to_base64(image_np, density_np, alpha=0.5):
    """Create heatmap overlay and return as base64 PNG."""
    overlay = create_overlay(image_np, density_np, alpha=alpha)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(np.clip(overlay, 0, 1))
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


def circles_to_base64(image_np, density_np):
    """Create circle-based head detection and return as base64 PNG."""
    fig, num_detected = draw_head_circles(image_np, density_np)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8'), num_detected


def numpy_to_base64(arr, cmap='jet', is_overlay=False):
    """Convert numpy array to base64 PNG via matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if is_overlay:
        ax.imshow(np.clip(arr, 0, 1))
    else:
        ax.imshow(arr, cmap=cmap)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=90)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def draw_points_on_image(image_np, points, color='red', size=4):
    """
    Draw points on a numpy image.
    Args:
        image_np: (H, W, 3) numpy array [0, 1]
        points:   (N, 2) list of [x, y] coordinates
        color:    Matplotlib color
        size:     Marker size
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image_np)
    if len(points) > 0:
        pts = np.array(points)
        ax.plot(pts[:, 0], pts[:, 1], 'o', color=color, markersize=size, 
                markeredgecolor='white', markeredgewidth=0.5)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def points_to_base64(image_np, points, color='red', size=4):
    """Render points on image and return as base64."""
    fig = draw_points_on_image(image_np, points, color=color, size=size)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=90)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
