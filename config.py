"""
CAN + CBAM Configuration — Centralized settings for unified training,
evaluation, and inference.

Single-model system: no per-part models, no routing logic.
Trained on merged ShanghaiTech Part A + Part B.
"""
import os
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = r"e:\crowd_density_estimator2\shanghaitech_dataset"

# ShanghaiTech raw data
SHANGHAITECH_ROOT = os.path.join(DATASET_ROOT, "ShanghaiTech")
PART_A_TRAIN = os.path.join(SHANGHAITECH_ROOT, "part_A", "train_data")
PART_A_TEST  = os.path.join(SHANGHAITECH_ROOT, "part_A", "test_data")
PART_B_TRAIN = os.path.join(SHANGHAITECH_ROOT, "part_B", "train_data")
PART_B_TEST  = os.path.join(SHANGHAITECH_ROOT, "part_B", "test_data")

# Preprocessed density maps (h5)
H5_ROOT = os.path.join(DATASET_ROOT, "shanghaitech_h5_empty", "ShanghaiTech")
H5_PART_A_TRAIN = os.path.join(H5_ROOT, "part_A", "train_data")
H5_PART_A_TEST  = os.path.join(H5_ROOT, "part_A", "test_data")
H5_PART_B_TRAIN = os.path.join(H5_ROOT, "part_B", "train_data")
H5_PART_B_TEST  = os.path.join(H5_ROOT, "part_B", "test_data")

# Negative samples directory (images with NO people)
NEGATIVE_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "negative_samples")

# Output directories
CHECKPOINT_DIR    = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR           = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR       = os.path.join(PROJECT_ROOT, "results")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualizations")
WEBAPP_DIR        = os.path.join(PROJECT_ROOT, "static")

for d in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, VISUALIZATION_DIR,
          WEBAPP_DIR, NEGATIVE_SAMPLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0   # Windows compatibility
PIN_MEMORY  = torch.cuda.is_available()

# ──────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
LEARNING_RATE     = 1e-4
BATCH_SIZE        = 4
EPOCHS            = 130
WEIGHT_DECAY      = 1e-4
FRONTEND_LR_SCALE = 0.1      # Frontend LR = base_lr × this

# Learning rate scheduler
LR_STEP_SIZE      = 30       # Decay LR every N epochs
LR_GAMMA          = 0.5      # Multiply LR by this factor

# Early stopping
EARLY_STOP_PATIENCE = 25

# Validation frequency (lazy validation for speed)
VALIDATE_EVERY    = 5         # Validate every N epochs
SUBSET_VAL_SIZE   = 50        # Number of images per test set for fast validation

# ──────────────────────────────────────────────────────────────────────────────
# Density Map Generation
# ──────────────────────────────────────────────────────────────────────────────
SIGMA_FIXED       = 15.0      # Fixed sigma for Part B
SIGMA_K           = 3         # k-nearest neighbors for adaptive sigma (Part A)
SIGMA_BETA        = 0.3       # Scaling factor for adaptive sigma

# ──────────────────────────────────────────────────────────────────────────────
# Data Augmentation & Training
# ──────────────────────────────────────────────────────────────────────────────
CROP_SIZE          = 256      # Random crop size for patch-based training
DOWNSAMPLE         = 8        # Density map downscale factor (matches CNN stride)

# Balanced sampling threshold (count above this = "dense")
DENSE_THRESHOLD    = 100      # Images with count > this are labeled "dense"

# ──────────────────────────────────────────────────────────────────────────────
# ImageNet normalization
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# CBAM Configuration
# ──────────────────────────────────────────────────────────────────────────────
CBAM_REDUCTION_RATIO = 16     # Channel attention MLP reduction ratio
CBAM_KERNEL_SIZE     = 7      # Spatial attention conv kernel size

# ──────────────────────────────────────────────────────────────────────────────
# CAN-specific Configuration
# ──────────────────────────────────────────────────────────────────────────────
CAN_DILATIONS         = (1, 2, 4, 12)   # Enhanced dilations for sparse crowds
USE_AUX_CLASSIFIER    = True            # Enable auxiliary dense/sparse head
AUX_LOSS_WEIGHT       = 0.1             # Weight for auxiliary classification loss
SSIM_LOSS_WEIGHT      = 0.001           # Weight for structural similarity loss

# ──────────────────────────────────────────────────────────────────────────────
# Segmentation Configuration
# ──────────────────────────────────────────────────────────────────────────────
USE_SEGMENTATION      = True    # Enable/disable human segmentation preprocessing
SEG_PERSON_CLASS      = 15      # COCO person class index for DeepLabV3
SEG_SOFT_MASK_FLOOR   = 0.1     # Minimum mask value (prevent complete suppression)

# ──────────────────────────────────────────────────────────────────────────────
# Gradient Clipping
# ──────────────────────────────────────────────────────────────────────────────
GRAD_CLIP_MAX_NORM    = 10.0

# ──────────────────────────────────────────────────────────────────────────────
# Convenience: get paths for a given part
# ──────────────────────────────────────────────────────────────────────────────
def get_paths(part: str):
    """Return (raw_train, raw_test, h5_train, h5_test) for part 'A' or 'B'."""
    part = part.upper()
    if part == "A":
        return PART_A_TRAIN, PART_A_TEST, H5_PART_A_TRAIN, H5_PART_A_TEST
    elif part == "B":
        return PART_B_TRAIN, PART_B_TEST, H5_PART_B_TRAIN, H5_PART_B_TEST
    else:
        raise ValueError(f"Unknown part: {part}. Use 'A' or 'B'.")


def print_config():
    """Print current configuration summary."""
    print("=" * 65)
    print("  CAN + CBAM — Configuration")
    print("=" * 65)
    print(f"  Device:            {DEVICE}")
    print(f"  Dataset root:      {DATASET_ROOT}")
    print(f"  LR:                {LEARNING_RATE}")
    print(f"  Frontend LR scale: {FRONTEND_LR_SCALE}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Crop size:         {CROP_SIZE}")
    print(f"  Validate every:    {VALIDATE_EVERY} epochs")
    print(f"  Subset val size:   {SUBSET_VAL_SIZE}")
    print(f"  CAN dilations:     {CAN_DILATIONS}")
    print(f"  Aux classifier:    {USE_AUX_CLASSIFIER}")
    print(f"  Segmentation:      {USE_SEGMENTATION}")
    print(f"  CBAM reduction:    {CBAM_REDUCTION_RATIO}")
    print(f"  CBAM kernel:       {CBAM_KERNEL_SIZE}")
    print("=" * 65)


if __name__ == "__main__":
    print_config()
