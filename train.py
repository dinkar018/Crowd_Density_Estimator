"""
Training Pipeline — Unified CAN+CBAM training on merged ShanghaiTech dataset.

Key differences from CSRNet system:
    - SINGLE model trained on merged Part A + Part B
    - Balanced sampling via WeightedRandomSampler
    - Lazy validation every N epochs (configurable)
    - Subset validation for speed
    - Auxiliary dense/sparse classification loss
    - Per-part evaluation metrics (Part A MAE, Part B MAE)

Features:
    - CAN backbone with multi-scale parallel dilated convolutions
    - CBAM attention at 3 positions
    - Optional human segmentation preprocessing
    - MSE loss on density maps + BCE loss for aux classifier
    - Adam optimizer with StepLR scheduling + frontend LR scaling
    - Early stopping, gradient clipping, model checkpointing
    - TensorBoard logging
"""
import os
import sys

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from model import CANCBAM, print_model_summary
from dataset import get_merged_train_loader, get_test_loader
from utils import compute_metrics, compute_ssim_psnr, plot_training_curves


# ──────────────────────────────────────────────────────────────────────────────
# Advanced Loss Functions
# ──────────────────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss for density map refinement.
    
    Why: MSE is poor at capturing the 'shapes' of density blobs in sparse crowds.
    SSIM enforces structural consistency, helping localize individual heads.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self._create_window(window_size, sigma)

    def _create_window(self, window_size, sigma):
        coords = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss /= gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return nn.Parameter(_2D_window, requires_grad=False)

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window.data = self.window.data.to(img1.device)
            
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class HybridLoss(nn.Module):
    """
    Loss Combination: MSE (Count) + SSIM (Structure) + BCE (Scene Type).
    """
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.ssim = SSIMLoss().to(device)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_density, gt_density, aux_logit, is_dense):
        # Handle size mismatch by resizing to match
        if pred_density.shape != gt_density.shape:
            # Resize the smaller tensor to match the larger one
            pred_h, pred_w = pred_density.shape[2], pred_density.shape[3]
            gt_h, gt_w = gt_density.shape[2], gt_density.shape[3]
            
            if pred_h < gt_h or pred_w < gt_w:
                # Resize pred to match gt
                pred_density = F.interpolate(pred_density, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
            elif gt_h < pred_h or gt_w < pred_w:
                # Resize gt to match pred
                gt_density = F.interpolate(gt_density, size=(pred_h, pred_w), mode='bilinear', align_corners=False)
        
        # 1. Density MSE Loss (normalized)
        density_loss = self.mse(pred_density, gt_density) / (pred_density.shape[0] * 1000)
        
        # 2. Structural SSIM Loss
        ssim_loss = self.ssim(pred_density, gt_density)
        
        # 3. Auxiliary Scene Classification Loss
        aux_loss = 0.0
        if aux_logit is not None:
            # Reshape is_dense to match aux_logit shape [B, 1]
            is_dense_t = is_dense.view(aux_logit.shape).to(pred_density.device)
            aux_loss = self.bce(aux_logit, is_dense_t)
            
        # Weighted combination
        total = (density_loss + 
                 config.SSIM_LOSS_WEIGHT * ssim_loss + 
                 config.AUX_LOSS_WEIGHT * aux_loss)
        
        return total, density_loss, ssim_loss, aux_loss


class UnifiedTrainer:
    """End-to-end unified training manager for CAN+CBAM."""

    def __init__(self, lr: float = None, batch_size: int = None,
                 epochs: int = None, resume: str = None,
                 use_segmentation: bool = None,
                 validate_every: int = None,
                 subset_val: int = None):
        self.lr = lr or config.LEARNING_RATE
        self.batch_size = batch_size or config.BATCH_SIZE
        self.epochs = epochs or config.EPOCHS
        self.device = config.DEVICE
        self.use_segmentation = (use_segmentation
                                 if use_segmentation is not None
                                 else config.USE_SEGMENTATION)
        self.validate_every = validate_every or config.VALIDATE_EVERY
        self.subset_val = subset_val or config.SUBSET_VAL_SIZE

        # Run name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        seg_tag = "_seg" if self.use_segmentation else ""
        self.run_name = f"CAN_CBAM_unified{seg_tag}_{timestamp}"
        self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, self.run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ── Model ──
        print(f"\n{'='*65}")
        print(f"  Initializing CAN + CBAM — Unified Model")
        print(f"  Device: {self.device}")
        print(f"  Segmentation: {'ON' if self.use_segmentation else 'OFF'}")
        print(f"  Validate every: {self.validate_every} epochs")
        print(f"  Subset validation: {self.subset_val} images per part")
        print(f"{'='*65}")

        self.model = CANCBAM(
            pretrained=True,
            reduction_ratio=config.CBAM_REDUCTION_RATIO,
            cbam_kernel=config.CBAM_KERNEL_SIZE,
            dilations=config.CAN_DILATIONS,
            use_aux_classifier=config.USE_AUX_CLASSIFIER,
        ).to(self.device)
        print_model_summary(self.model)

        # ── Segmentation Module (optional) ──
        self.segmentor = None
        if self.use_segmentation:
            from segmentation import HumanSegmentor
            self.segmentor = HumanSegmentor(device=self.device)

        # ── Optimizer with frontend LR scaling ──
        param_groups = self.model.get_param_groups(
            frontend_lr_scale=config.FRONTEND_LR_SCALE
        )
        self.optimizer = torch.optim.Adam([
            {'params': param_groups[0]['params'],
             'lr': self.lr * param_groups[0]['lr_scale']},
            {'params': param_groups[1]['params'],
             'lr': self.lr * param_groups[1]['lr_scale']},
        ], weight_decay=config.WEIGHT_DECAY)

        # ── LR Scheduler ──
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA,
        )

        # ── Loss ──
        self.criterion = HybridLoss(self.device).to(self.device)

        # ── Data: Merged training, separate test ──
        self.train_loader, self.dataset_info = get_merged_train_loader(
            self.batch_size,
            negative_samples_dir=config.NEGATIVE_SAMPLES_DIR,
        )
        self.test_loader_A, self.test_dataset_A = get_test_loader("A")
        self.test_loader_B, self.test_dataset_B = get_test_loader("B")

        # ── TensorBoard ──
        log_path = os.path.join(config.LOG_DIR, self.run_name)
        self.writer = SummaryWriter(log_dir=log_path)

        # ── Tracking ──
        self.best_combined_mae = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [], 'lr': [],
            'val_loss_A': [], 'val_mae_A': [], 'val_rmse_A': [], 'val_r2_A': [],
            'val_loss_B': [], 'val_mae_B': [], 'val_rmse_B': [], 'val_r2_B': [],
            'aux_accuracy': [],
        }
        self.start_epoch = 0

        # Resume from checkpoint
        if resume and os.path.exists(resume):
            self._load_checkpoint(resume)

        # cuDNN benchmark
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"  CUDA: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Memory: {mem:.1f} GB")

    def _apply_segmentation(self, images):
        """Apply human segmentation mask to a batch of images."""
        if self.segmentor is None:
            return images
        return self.segmentor.apply_mask(images)

    def train_epoch(self, epoch):
        """Train for one epoch on merged dataset."""
        self.model.train()
        epoch_loss = 0.0
        epoch_density_loss = 0.0
        epoch_ssim_loss = 0.0
        epoch_aux_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader,
                     desc=f"  Epoch {epoch+1:3d}/{self.epochs}",
                     ncols=110, leave=True)

        for batch in pbar:
            images, densities, counts, is_dense = batch

            if isinstance(images, list):
                # Variable-size batch (shouldn't happen in training, but handle)
                batch_loss = torch.tensor(0.0, device=self.device,
                                          requires_grad=True)
                for img, den, dense_label in zip(images, densities, is_dense):
                    img = img.unsqueeze(0).to(self.device)
                    den = den.unsqueeze(0).to(self.device)
                    img = self._apply_segmentation(img)

                    result = self.model(img)
                    if isinstance(result, tuple):
                        pred, aux_logit = result
                    else:
                        pred = result
                        aux_logit = None

                    loss = self.criterion_density(pred, den) / 1000
                    if aux_logit is not None:
                        dense_t = dense_label.unsqueeze(0).unsqueeze(0).to(
                            self.device)
                        aux_loss = self.criterion_aux(aux_logit, dense_t)
                        loss = loss + config.AUX_LOSS_WEIGHT * aux_loss

                    batch_loss = batch_loss + loss
                batch_loss = batch_loss / len(images)
            else:
                images = images.to(self.device)
                densities = densities.to(self.device)
                images = self._apply_segmentation(images)

                result = self.model(images)
                if isinstance(result, tuple):
                    pred_density, aux_logit = result
                else:
                    pred_density = result
                    aux_logit = None

                # Use the new HybridLoss
                batch_loss, d_loss, s_loss, a_loss = self.criterion(
                    pred_density, densities, aux_logit, is_dense
                )

                epoch_density_loss += d_loss.item()
                epoch_ssim_loss += s_loss.item() if isinstance(s_loss, torch.Tensor) else s_loss
                epoch_aux_loss += a_loss.item() if isinstance(a_loss, torch.Tensor) else a_loss

            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            max_norm=config.GRAD_CLIP_MAX_NORM)
            self.optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches += 1
            pbar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "mse": f"{d_loss.item():.4f}",
                "ssim": f"{s_loss.item():.3f}" if isinstance(s_loss, torch.Tensor) else f"{s_loss:.3f}"
            })

        avg_loss = epoch_loss / max(n_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, part: str, subset_size: int = None):
        """
        Validate on a specific test set.

        Args:
            part:        'A' or 'B'
            subset_size: Number of images to validate on (None = all)

        Returns:
            metrics: dict with MAE, RMSE, R², etc.
        """
        self.model.eval()
        subset_size = subset_size or self.subset_val

        if part == "A":
            test_dataset = self.test_dataset_A
        else:
            test_dataset = self.test_dataset_B

        # Use subset for speed
        n_samples = min(subset_size, len(test_dataset))
        indices = list(range(n_samples))

        pred_counts = []
        gt_counts = []
        val_loss = 0.0
        aux_correct = 0
        aux_total = 0

        for idx in tqdm(indices, desc=f"Val Part {part}", leave=True):
            img_tensor, gt_density, gt_count_tensor, is_dense = test_dataset[idx]
            gt_count = gt_count_tensor.item()

            img_input = img_tensor.unsqueeze(0).to(self.device); img_input = torch.nn.functional.interpolate(img_input, size=(768, 1024), mode='bilinear', align_corners=False) if img_input.shape[2] > 768 or img_input.shape[3] > 1024 else img_input
            gt_density_d = gt_density.unsqueeze(0).to(self.device)
            img_input = self._apply_segmentation(img_input)

            result = self.model(img_input)
            if isinstance(result, tuple):
                pred_density, aux_logit = result
            else:
                pred_density = result
                aux_logit = None

            # Use the new HybridLoss (val_loss is mainly density loss)
            loss_t, density_l, ssim_l, aux_l = self.criterion(
                pred_density, gt_density_d, aux_logit, is_dense
            )
            val_loss += density_l.item()

            pred_count = pred_density.sum().item()
            pred_counts.append(pred_count)
            gt_counts.append(gt_count)

            # Aux classifier accuracy
            if aux_logit is not None:
                pred_dense = (torch.sigmoid(aux_logit) > 0.5).float().item()
                aux_correct += int(pred_dense == is_dense.item())
                aux_total += 1

        pred_counts = np.array(pred_counts)
        gt_counts = np.array(gt_counts)

        metrics = compute_metrics(pred_counts, gt_counts)
        metrics['val_loss'] = val_loss / max(n_samples, 1)
        if aux_total > 0:
            metrics['aux_accuracy'] = aux_correct / aux_total

        return metrics

    def train(self):
        """Full training loop."""
        print(f"\n{'='*65}")
        print(f"  Training CAN + CBAM — Unified Model")
        print(f"  LR={self.lr}, Batch={self.batch_size}, Epochs={self.epochs}")
        print(f"  Segmentation: {'ENABLED' if self.use_segmentation else 'DISABLED'}")
        print(f"  CAN dilations: {config.CAN_DILATIONS}")
        print(f"  Aux classifier: {config.USE_AUX_CLASSIFIER}")
        print(f"  Validate every: {self.validate_every} epochs")
        print(f"{'='*65}\n")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()

            # ── Train ──
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - t0

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(current_lr)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)

            # ── Lazy Validation ──
            do_validate = (
                (epoch + 1) % self.validate_every == 0
                or epoch == 0
                or epoch == self.epochs - 1
            )

            if do_validate:
                metrics_a = self.validate("A")
                metrics_b = self.validate("B")

                # Log Part A metrics
                self.history['val_loss_A'].append(metrics_a['val_loss'])
                self.history['val_mae_A'].append(metrics_a['MAE'])
                self.history['val_rmse_A'].append(metrics_a['RMSE'])
                self.history['val_r2_A'].append(metrics_a['R2'])

                # Log Part B metrics
                self.history['val_loss_B'].append(metrics_b['val_loss'])
                self.history['val_mae_B'].append(metrics_b['MAE'])
                self.history['val_rmse_B'].append(metrics_b['RMSE'])
                self.history['val_r2_B'].append(metrics_b['R2'])

                # Aux accuracy
                if 'aux_accuracy' in metrics_a:
                    avg_aux = (metrics_a.get('aux_accuracy', 0) +
                               metrics_b.get('aux_accuracy', 0)) / 2
                    self.history['aux_accuracy'].append(avg_aux)

                # TensorBoard
                self.writer.add_scalar('Metrics/MAE_A', metrics_a['MAE'],
                                       epoch)
                self.writer.add_scalar('Metrics/MAE_B', metrics_b['MAE'],
                                       epoch)
                self.writer.add_scalar('Metrics/RMSE_A', metrics_a['RMSE'],
                                       epoch)
                self.writer.add_scalar('Metrics/RMSE_B', metrics_b['RMSE'],
                                       epoch)

                # Combined MAE (average of Part A and Part B)
                combined_mae = (metrics_a['MAE'] + metrics_b['MAE']) / 2

                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train: {train_loss:.4f} | "
                      f"MAE [A: {metrics_a['MAE']:.2f} / "
                      f"B: {metrics_b['MAE']:.2f}] | "
                      f"Combined: {combined_mae:.2f} | "
                      f"LR: {current_lr:.1e} | "
                      f"{epoch_time:.1f}s")

                # Checkpointing based on combined MAE
                is_best = combined_mae < self.best_combined_mae
                if is_best:
                    self.best_combined_mae = combined_mae
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, metrics_a, metrics_b,
                                          is_best=True)
                    print(f"  ★ New best combined MAE: {self.best_combined_mae:.2f} "
                          f"(A: {metrics_a['MAE']:.2f}, B: {metrics_b['MAE']:.2f})")
                else:
                    self.epochs_without_improvement += self.validate_every
            else:
                # Non-validation epoch
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train: {train_loss:.4f} | "
                      f"LR: {current_lr:.1e} | "
                      f"{epoch_time:.1f}s")

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, {}, {}, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} "
                      f"(no improvement for "
                      f"{config.EARLY_STOP_PATIENCE} epochs)")
                break

        self.writer.close()
        self._save_history()
        self._print_summary()

        # Plot training curves
        plot_training_curves(self.history, self.checkpoint_dir)

        return self.best_combined_mae

    def _save_checkpoint(self, epoch, metrics_a, metrics_b, is_best=False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_combined_mae': self.best_combined_mae,
            'metrics_A': metrics_a,
            'metrics_B': metrics_b,
            'architecture': 'CAN_CBAM',
            'use_segmentation': self.use_segmentation,
            'config': {
                'lr': self.lr,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'dilations': config.CAN_DILATIONS,
                'cbam_reduction': config.CBAM_REDUCTION_RATIO,
                'cbam_kernel': config.CBAM_KERNEL_SIZE,
                'use_aux': config.USE_AUX_CLASSIFIER,
            },
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, path)
            # Also save top-level best model
            top_path = os.path.join(config.CHECKPOINT_DIR,
                                     'best_model_unified.pth')
            torch.save(state, top_path)
        else:
            path = os.path.join(self.checkpoint_dir,
                                 f'checkpoint_epoch{epoch+1}.pth')
            torch.save(state, path)

    def _load_checkpoint(self, path):
        """Resume training from checkpoint."""
        print(f"  Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_combined_mae = checkpoint.get('best_combined_mae', float('inf'))
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"  Resumed from epoch {self.start_epoch}, "
              f"best combined MAE: {self.best_combined_mae:.2f}")

    def _save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"  History saved: {path}")

    def _print_summary(self):
        """Print final training summary."""
        print(f"\n{'='*65}")
        print(f"  Training Complete — CAN + CBAM Unified Model")
        print(f"{'='*65}")
        print(f"  Best combined MAE: {self.best_combined_mae:.2f}")
        if self.history['val_mae_A']:
            best_idx = np.argmin(
                [a + b for a, b in zip(self.history['val_mae_A'],
                                        self.history['val_mae_B'])]
            )
            print(f"  Best Part A MAE:   {self.history['val_mae_A'][best_idx]:.2f}")
            print(f"  Best Part A RMSE:  {self.history['val_rmse_A'][best_idx]:.2f}")
            print(f"  Best Part B MAE:   {self.history['val_mae_B'][best_idx]:.2f}")
            print(f"  Best Part B RMSE:  {self.history['val_rmse_B'][best_idx]:.2f}")
        print(f"  Segmentation:      {'ENABLED' if self.use_segmentation else 'DISABLED'}")
        print(f"  Checkpoints:       {self.checkpoint_dir}")
        print(f"{'='*65}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CAN + CBAM (Unified)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--no_seg", action="store_true",
                        help="Disable segmentation preprocessing")
    parser.add_argument("--validate_every", type=int, default=None,
                        help="Validate every N epochs")
    parser.add_argument("--subset_val", type=int, default=None,
                        help="Number of images per part for validation")
    args = parser.parse_args()

    trainer = UnifiedTrainer(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        resume=args.resume,
        use_segmentation=not args.no_seg,
        validate_every=args.validate_every,
        subset_val=args.subset_val,
    )
    trainer.train()
