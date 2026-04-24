"""
Testing Pipeline — Complete inference and evaluation with comprehensive
visualization outputs for CAN+CBAM unified model.

Evaluates the SINGLE model on both Part A and Part B test sets separately.

For each test image, generates and saves:
    1. Original image
    2. Segmentation mask
    3. Ground truth density map
    4. Predicted density map
    5. Heatmap overlay on original image
    6. Circle-based head detection
    7. Error map (absolute difference)

Also computes and reports MAE, RMSE per dataset.
"""
import os
import sys

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from model import CANCBAM
from dataset import get_test_loader, CrowdDataset
from utils import (
    compute_metrics, compute_ssim_psnr,
    visualize_prediction_full, denormalize_image,
    detect_peaks, get_adaptive_radius, GradCAM,
)
from PIL import Image


class UnifiedTester:
    """Full testing suite for trained CAN+CBAM unified model."""

    def __init__(self, checkpoint_path: str = None,
                 use_segmentation: bool = None):
        self.device = config.DEVICE
        self.use_segmentation = (use_segmentation
                                 if use_segmentation is not None
                                 else config.USE_SEGMENTATION)

        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR,
                                            'best_model_unified.pth')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\n{'='*65}")
        print(f"  Testing CAN + CBAM — Unified Model")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Segmentation: {'ON' if self.use_segmentation else 'OFF'}")
        print(f"{'='*65}")

        # Load model
        self.model = CANCBAM(
            pretrained=False,
            dilations=config.CAN_DILATIONS,
            use_aux_classifier=config.USE_AUX_CLASSIFIER,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"  Best combined MAE: "
              f"{checkpoint.get('best_combined_mae', 'N/A')}")
        print(f"  Trained for epochs: {checkpoint.get('epoch', 'N/A')}")

        # Segmentation module
        self.segmentor = None
        if self.use_segmentation:
            from segmentation import HumanSegmentor
            self.segmentor = HumanSegmentor(device=self.device)

        # Test DataLoaders + Datasets
        _, self.test_dataset_A = get_test_loader("A")
        _, self.test_dataset_B = get_test_loader("B")

        # Output directory
        self.output_dir = os.path.join(config.RESULTS_DIR, 'test_unified')
        os.makedirs(self.output_dir, exist_ok=True)

    def _apply_segmentation(self, images, return_mask=False):
        """Apply human segmentation mask if enabled."""
        if self.segmentor is None:
            if return_mask:
                return images, None
            return images
        return self.segmentor.apply_mask(images, return_mask=return_mask)

    @torch.no_grad()
    def evaluate(self, save_visualizations: bool = True, max_vis: int = 20,
                 run_gradcam: bool = False):
        """
        Run full evaluation on both Part A and Part B test sets.

        Args:
            save_visualizations: Whether to save per-image visual outputs
            max_vis:             Max images to visualize per part
            run_gradcam:         Whether to run Grad-CAM visualization
        """
        all_metrics = {}

        for part, test_dataset in [("A", self.test_dataset_A),
                                    ("B", self.test_dataset_B)]:
            print(f"\n  Evaluating Part {part} ({len(test_dataset)} images)...")

            vis_dir = os.path.join(self.output_dir, f'part_{part}',
                                    'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            results = []
            pred_counts_all = []
            gt_counts_all = []
            ssim_scores = []
            psnr_scores = []
            vis_count = 0

            for idx in tqdm(range(len(test_dataset)),
                            desc=f"  Part {part}", ncols=100):
                img_tensor, gt_density, gt_count_tensor, is_dense = \
                    test_dataset[idx]
                gt_count = gt_count_tensor.item()

                # Inference
                img_input = img_tensor.unsqueeze(0).to(self.device)

                mask_np = None
                if self.segmentor is not None:
                    img_input, mask = self._apply_segmentation(
                        img_input, return_mask=True
                    )
                    mask_np = mask.squeeze().cpu().numpy()
                else:
                    img_input = self._apply_segmentation(img_input)

                result = self.model(img_input)
                if isinstance(result, tuple):
                    pred_density, aux_logit = result
                else:
                    pred_density = result

                pred_count = pred_density.sum().item()
                error = abs(pred_count - gt_count)

                # SSIM/PSNR
                gt_density_dev = gt_density.unsqueeze(0).to(self.device)
                s, p = compute_ssim_psnr(pred_density, gt_density_dev)

                results.append({
                    'index': idx,
                    'image': os.path.basename(
                        test_dataset.image_paths[idx]
                    ) if idx < len(test_dataset.image_paths) else f"img_{idx}",
                    'gt_count': gt_count,
                    'pred_count': pred_count,
                    'abs_error': error,
                    'rel_error': error / max(gt_count, 1) * 100,
                    'ssim': s,
                    'psnr': p,
                    'part': part,
                })

                pred_counts_all.append(pred_count)
                gt_counts_all.append(gt_count)
                ssim_scores.append(s)
                psnr_scores.append(p)

                # ── Save full 7-panel visualization ──
                if save_visualizations and vis_count < max_vis:
                    save_path = os.path.join(vis_dir, f"test_{idx:03d}.png")
                    visualize_prediction_full(
                        img_tensor,
                        gt_density,
                        pred_density.squeeze(0).cpu(),
                        gt_count,
                        pred_count,
                        mask_np=mask_np,
                        save_path=save_path,
                        title=f"CAN+CBAM Unified — Part {part} Test {idx}"
                    )
                    vis_count += 1

            pred_counts_all = np.array(pred_counts_all)
            gt_counts_all = np.array(gt_counts_all)

            # Aggregate metrics
            metrics = compute_metrics(pred_counts_all, gt_counts_all)
            metrics['SSIM'] = float(np.mean(ssim_scores))
            metrics['PSNR'] = float(np.mean(psnr_scores))
            metrics['num_images'] = len(results)

            all_metrics[part] = metrics

            # Print results
            self._print_results(part, metrics)

            # Save results
            self._save_results(part, results, metrics)

            # Plots
            self._plot_scatter(part, pred_counts_all, gt_counts_all)
            self._plot_error_distribution(part, pred_counts_all, gt_counts_all)

        # ── Print combined summary ──
        self._print_combined_summary(all_metrics)

        # ── Grad-CAM (optional) ──
        if run_gradcam:
            self._run_gradcam()

        return all_metrics

    def _print_results(self, part, metrics):
        """Print formatted evaluation results for a single part."""
        print(f"\n{'='*65}")
        print(f"  CAN + CBAM Unified — Part {part} Results")
        print(f"{'='*65}")
        print(f"  {'Metric':<20} {'Value':>12}")
        print(f"  {'-'*32}")
        print(f"  {'MAE':<20} {metrics['MAE']:>12.2f}")
        print(f"  {'RMSE':<20} {metrics['RMSE']:>12.2f}")
        print(f"  {'MSE':<20} {metrics['MSE']:>12.2f}")
        print(f"  {'MAPE (%)':<20} {metrics['MAPE']:>12.2f}")
        print(f"  {'R²':<20} {metrics['R2']:>12.4f}")
        print(f"  {'SSIM':<20} {metrics['SSIM']:>12.4f}")
        print(f"  {'PSNR (dB)':<20} {metrics['PSNR']:>12.2f}")
        print(f"  {'Images':<20} {metrics['num_images']:>12}")
        print(f"{'='*65}")

    def _print_combined_summary(self, all_metrics):
        """Print combined summary across both parts."""
        print(f"\n{'='*65}")
        print(f"  CAN + CBAM — Combined Summary")
        print(f"{'='*65}")
        if 'A' in all_metrics and 'B' in all_metrics:
            avg_mae = (all_metrics['A']['MAE'] + all_metrics['B']['MAE']) / 2
            avg_rmse = (all_metrics['A']['RMSE'] + all_metrics['B']['RMSE']) / 2
            print(f"  Part A MAE:        {all_metrics['A']['MAE']:.2f}")
            print(f"  Part B MAE:        {all_metrics['B']['MAE']:.2f}")
            print(f"  Combined MAE:      {avg_mae:.2f}")
            print(f"  Part A RMSE:       {all_metrics['A']['RMSE']:.2f}")
            print(f"  Part B RMSE:       {all_metrics['B']['RMSE']:.2f}")
            print(f"  Combined RMSE:     {avg_rmse:.2f}")
        print(f"{'='*65}\n")

    def _save_results(self, part, results, metrics):
        """Save per-image results and summary."""
        import pandas as pd

        part_dir = os.path.join(self.output_dir, f'part_{part}')
        os.makedirs(part_dir, exist_ok=True)

        df = pd.DataFrame(results)
        csv_path = os.path.join(part_dir, 'per_image_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Per-image results: {csv_path}")

        json_path = os.path.join(part_dir, 'evaluation_summary.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Summary: {json_path}")

    def _plot_scatter(self, part, pred_counts, gt_counts):
        """Scatter plot: Predicted vs. Actual counts."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.scatter(gt_counts, pred_counts, alpha=0.6, edgecolors='navy',
                   facecolors='dodgerblue', s=40, linewidths=0.5)

        max_val = max(gt_counts.max(), pred_counts.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')

        if len(gt_counts) > 1:
            z = np.polyfit(gt_counts, pred_counts, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(0, max_val, 100)
            ax.plot(x_fit, p(x_fit), 'g-', lw=1.5, alpha=0.8,
                    label=f'Fit: y={z[0]:.2f}x+{z[1]:.1f}')

        ax.set_xlabel('Ground Truth Count', fontsize=14)
        ax.set_ylabel('Predicted Count', fontsize=14)
        ax.set_title(f'CAN+CBAM Unified — Part {part}', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        part_dir = os.path.join(self.output_dir, f'part_{part}')
        path = os.path.join(part_dir, 'scatter_pred_vs_actual.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Scatter plot: {path}")

    def _plot_error_distribution(self, part, pred_counts, gt_counts):
        """Error distribution histogram."""
        errors = pred_counts - gt_counts

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(np.abs(errors), bins=30, color='steelblue',
                     edgecolor='white', alpha=0.8)
        axes[0].axvline(np.abs(errors).mean(), color='red', linestyle='--',
                        label=f'Mean: {np.abs(errors).mean():.1f}')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Absolute Error Distribution')
        axes[0].legend()

        axes[1].hist(errors, bins=30, color='coral', edgecolor='white',
                     alpha=0.8)
        axes[1].axvline(0, color='black', linestyle='-', lw=1)
        axes[1].axvline(errors.mean(), color='red', linestyle='--',
                        label=f'Mean bias: {errors.mean():.1f}')
        axes[1].set_xlabel('Error (Pred - Actual)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Signed Error Distribution')
        axes[1].legend()

        fig.suptitle(f'CAN+CBAM Unified — Part {part} Error Analysis',
                     fontsize=16, y=1.02)
        part_dir = os.path.join(self.output_dir, f'part_{part}')
        path = os.path.join(part_dir, 'error_distribution.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Error distribution: {path}")

    def _run_gradcam(self, n_images=5):
        """Run Grad-CAM visualization on a few test images."""
        print(f"\n  Running Grad-CAM on {n_images} test images...")

        gradcam_dir = os.path.join(self.output_dir, 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)

        # Need to enable gradients for Grad-CAM
        self.model.eval()
        grad_cam = GradCAM(self.model)

        for part, test_dataset in [("A", self.test_dataset_A),
                                    ("B", self.test_dataset_B)]:
            for idx in range(min(n_images, len(test_dataset))):
                img_tensor, _, _, _ = test_dataset[idx]
                img_input = img_tensor.unsqueeze(0).to(self.device)
                img_input.requires_grad_(True)

                img_np = denormalize_image(img_tensor)

                save_path = os.path.join(gradcam_dir,
                                          f'gradcam_part{part}_{idx:03d}.png')
                grad_cam.visualize(img_input, img_np, save_path=save_path)

        print(f"  Grad-CAM saved to: {gradcam_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Single Image Inference
# ──────────────────────────────────────────────────────────────────────────────

def infer_single_image(image_path: str, checkpoint_path: str = None,
                       use_segmentation: bool = True,
                       save_dir: str = None):
    """
    Run inference on a single image and save all visual outputs.

    Args:
        image_path:       path to input image
        checkpoint_path:  path to model checkpoint
        use_segmentation: whether to apply segmentation mask
        save_dir:         directory to save outputs
    """
    device = config.DEVICE

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR,
                                        'best_model_unified.pth')

    # Load model
    model = CANCBAM(
        pretrained=False,
        dilations=config.CAN_DILATIONS,
        use_aux_classifier=config.USE_AUX_CLASSIFIER,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Segmentation
    segmentor = None
    mask_np = None
    if use_segmentation:
        from segmentation import HumanSegmentor
        segmentor = HumanSegmentor(device=device)

    # Load and prepare image
    from utils import prepare_image_for_inference
    pil_img = Image.open(image_path).convert('RGB')
    img_tensor, original_size = prepare_image_for_inference(pil_img, device)

    # Apply segmentation
    if segmentor is not None:
        img_tensor, mask = segmentor.apply_mask(img_tensor, return_mask=True)
        mask_np = mask.squeeze().cpu().numpy()

    # Inference
    with torch.no_grad():
        result = model(img_tensor)
        if isinstance(result, tuple):
            pred_density, aux_logit = result
            is_dense = torch.sigmoid(aux_logit).item() > 0.5
            print(f"  Scene type: {'Dense' if is_dense else 'Sparse'}")
        else:
            pred_density = result

    pred_count = pred_density.sum().item()

    print(f"\n  Image: {os.path.basename(image_path)}")
    print(f"  Predicted count: {pred_count:.1f}")

    # Save visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            os.path.basename(image_path).replace('.jpg', '_result.png')
        )

        pred_np = pred_density.squeeze().cpu()
        dummy_gt = torch.zeros_like(pred_np)

        visualize_prediction_full(
            img_tensor.squeeze(0).cpu(),
            dummy_gt.unsqueeze(0),
            pred_np.unsqueeze(0),
            0, pred_count,
            mask_np=mask_np,
            save_path=save_path,
            title=f"CAN+CBAM Inference: {os.path.basename(image_path)}"
        )
        print(f"  Saved: {save_path}")

    return pred_count


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test CAN + CBAM Unified")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path for inference")
    parser.add_argument("--no_seg", action="store_true",
                        help="Disable segmentation preprocessing")
    parser.add_argument("--max_vis", type=int, default=20,
                        help="Max images to visualize per part")
    parser.add_argument("--gradcam", action="store_true",
                        help="Run Grad-CAM visualization")
    args = parser.parse_args()

    if args.image:
        infer_single_image(
            args.image,
            checkpoint_path=args.checkpoint,
            use_segmentation=not args.no_seg,
            save_dir=os.path.join(config.RESULTS_DIR, 'single_inference')
        )
    else:
        tester = UnifiedTester(
            checkpoint_path=args.checkpoint,
            use_segmentation=not args.no_seg,
        )
        tester.evaluate(
            save_visualizations=True,
            max_vis=args.max_vis,
            run_gradcam=args.gradcam,
        )
