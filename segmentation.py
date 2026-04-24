"""
Human Segmentation Preprocessing — Suppresses non-human regions before
feeding images to CAN+CBAM, reducing false positives from trees,
buildings, and other background clutter.

Uses a pretrained DeepLabV3 (ResNet-101 backbone) to extract person-class
masks and applies them as soft attention masks on the input image.

Pipeline:
    Raw Image → DeepLabV3 → Person Mask → Soft Mask (floor=0.1) → Masked Image

The soft mask ensures:
    - Human regions: fully preserved (mask ≈ 1.0)
    - Background regions: heavily suppressed but not zeroed (mask ≥ 0.1)
    - Prevents the model from hallucinating people in trees/textures
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

import config


class HumanSegmentor:
    """
    Human segmentation module using pretrained DeepLabV3-ResNet101.

    Extracts person-class masks from COCO-pretrained segmentation model
    and creates soft attention masks to suppress non-human regions.

    Usage:
        segmentor = HumanSegmentor(device='cuda')
        masked_image = segmentor.apply_mask(image_tensor)  # (B, 3, H, W)
    """

    # COCO class index for 'person'
    PERSON_CLASS = 15

    def __init__(self, device=None, soft_floor=None):
        """
        Initialize the segmentation model.

        Args:
            device:     torch device (default: from config)
            soft_floor: minimum mask value to prevent complete suppression
        """
        self.device = device or config.DEVICE
        self.soft_floor = soft_floor or config.SEG_SOFT_MASK_FLOOR

        # Load pretrained DeepLabV3 (ResNet-101 backbone, COCO)
        from torchvision.models.segmentation import deeplabv3_resnet101
        from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

        print("  Loading DeepLabV3-ResNet101 segmentation model...")
        self.model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT
        ).to(self.device)
        self.model.eval()

        # Freeze all segmentation parameters (inference only)
        for param in self.model.parameters():
            param.requires_grad = False

        # ImageNet normalization (same as DeepLabV3 expects)
        self.normalize = T.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        )

        print(f"  [OK] Segmentation model loaded on {self.device}")
        print(f"    Soft mask floor: {self.soft_floor}")

    @torch.no_grad()
    def get_person_mask(self, image_tensor):
        """
        Extract person segmentation mask from an image tensor.

        Args:
            image_tensor: (B, 3, H, W) normalized image tensor

        Returns:
            mask: (B, 1, H, W) binary person mask (0 or 1)
        """
        _, _, h, w = image_tensor.shape
        need_resize = h < 224 or w < 224

        if need_resize:
            scale = max(224 / h, 224 / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            input_tensor = F.interpolate(
                image_tensor, size=(new_h, new_w),
                mode='bilinear', align_corners=False
            )
        else:
            input_tensor = image_tensor

        # Run segmentation
        output = self.model(input_tensor)['out']  # (B, 21, H', W')
        predictions = output.argmax(dim=1)         # (B, H', W')

        # Extract person class mask
        person_mask = (predictions == self.PERSON_CLASS).float()
        person_mask = person_mask.unsqueeze(1)      # (B, 1, H', W')

        # Resize back to original resolution if needed
        if need_resize:
            person_mask = F.interpolate(
                person_mask, size=(h, w), mode='nearest'
            )

        return person_mask

    @torch.no_grad()
    def get_soft_mask(self, image_tensor):
        """
        Extract soft person mask (with floor value for background).

        Background is not completely zeroed — a floor value (default 0.1)
        is maintained so the density estimator still has some context.

        Args:
            image_tensor: (B, 3, H, W) normalized image tensor

        Returns:
            soft_mask: (B, 1, H, W) values in [soft_floor, 1.0]
        """
        binary_mask = self.get_person_mask(image_tensor)

        # Apply Gaussian blur to soften mask edges
        kernel_size = 11
        padding = kernel_size // 2
        soft_mask = F.avg_pool2d(
            binary_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Rescale: ensure human regions stay at 1.0
        if soft_mask.max() > 0:
            soft_mask = soft_mask / (soft_mask.max() + 1e-8)

        # Apply floor so background isn't completely black
        soft_mask = torch.clamp(soft_mask, min=self.soft_floor, max=1.0)

        return soft_mask

    @torch.no_grad()
    def apply_mask(self, image_tensor, return_mask=False):
        """
        Apply human segmentation mask to suppress non-human regions.

        Args:
            image_tensor: (B, 3, H, W) normalized image tensor
            return_mask:  if True, also return the mask

        Returns:
            masked_image: (B, 3, H, W) image with background suppressed
            mask:         (B, 1, H, W) soft mask (only if return_mask=True)
        """
        soft_mask = self.get_soft_mask(image_tensor)
        masked_image = image_tensor * soft_mask

        if return_mask:
            return masked_image, soft_mask
        return masked_image

    @torch.no_grad()
    def apply_mask_pil(self, pil_image):
        """
        Apply segmentation mask to a PIL image.
        Useful for inference/web app where we start with raw images.

        Args:
            pil_image: PIL Image (RGB)

        Returns:
            masked_tensor: (1, 3, H, W) masked and normalized image tensor
            mask_np:       (H, W) numpy mask for visualization
        """
        to_tensor = T.ToTensor()
        img_tensor = to_tensor(pil_image).unsqueeze(0)
        img_normalized = self.normalize(img_tensor.clone())
        img_normalized = img_normalized.to(self.device)

        # Get mask
        soft_mask = self.get_soft_mask(img_normalized)

        # Apply mask
        masked = img_normalized * soft_mask

        # Extract mask for visualization
        mask_np = soft_mask.squeeze().cpu().numpy()

        return masked, mask_np


def visualize_segmentation(image_path: str, output_path: str = None):
    """
    Visualize the segmentation mask for a single image.
    Useful for debugging and understanding mask quality.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert('RGB')
    segmentor = HumanSegmentor()

    masked_tensor, mask_np = segmentor.apply_mask_pil(img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Person Soft Mask', fontsize=14)
    axes[1].axis('off')

    # Reconstruct masked image for display
    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(config.IMAGENET_MEAN, config.IMAGENET_STD)],
        std=[1 / s for s in config.IMAGENET_STD]
    )
    masked_display = inv_normalize(masked_tensor.squeeze().cpu())
    masked_display = torch.clamp(masked_display, 0, 1)
    masked_display = masked_display.permute(1, 2, 0).numpy()

    axes[2].imshow(masked_display)
    axes[2].set_title('Masked Image (background suppressed)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    if output_path is None:
        output_path = image_path.replace('.jpg', '_segmentation.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Segmentation visualization saved: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_segmentation(sys.argv[1])
    else:
        print("Usage: python segmentation.py <image_path>")
        print("  Generates segmentation mask visualization for the given image.")
