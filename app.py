"""
Flask Web Application — Crowd Density Estimation with CAN+CBAM.

UNIFIED MODEL: No model selection needed — single model handles all scenes.

Features:
    - Upload an image
    - Run pipeline: segmentation → CAN+CBAM → output
    - Display: original, segmented, heatmap overlay, circle-based head
      visualization, predicted count

Usage:
    python app.py [--port 5000] [--no_seg]
"""
import os
import sys
import json

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import io
import base64
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config
from model import CANCBAM
from utils import (
    prepare_image_for_inference, denormalize_image,
    numpy_to_base64, circles_to_base64,
    detect_peaks, get_adaptive_radius,
)

# ──────────────────────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Global model and segmentor
model = None
model_info = {}
segmentor = None
device = config.DEVICE


def load_model(use_segmentation=True):
    """Load the unified CAN+CBAM model."""
    global model, model_info, segmentor

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR,
                                    'best_model_unified.pth')
    if not os.path.exists(checkpoint_path):
        print(f"  [WARN] Unified checkpoint not found: {checkpoint_path}")
        return

    print(f"  Loading CAN+CBAM unified model: {checkpoint_path}")
    model = CANCBAM(
        pretrained=False,
        dilations=config.CAN_DILATIONS,
        use_aux_classifier=config.USE_AUX_CLASSIFIER,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model_info = {
        'combined_mae': checkpoint.get('best_combined_mae', 'N/A'),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'metrics_A': checkpoint.get('metrics_A', {}),
        'metrics_B': checkpoint.get('metrics_B', {}),
        'architecture': 'CAN + CBAM',
        'dilations': str(config.CAN_DILATIONS),
    }

    mae = model_info['combined_mae']
    print(f"  [OK] Model loaded (Combined MAE: "
          f"{mae:.2f})" if isinstance(mae, (int, float)) else
          f"  [OK] Model loaded")

    if use_segmentation:
        try:
            from segmentation import HumanSegmentor
            segmentor = HumanSegmentor(device=device)
        except Exception as e:
            print(f"  [WARN] Segmentation unavailable: {e}")
            segmentor = None
    else:
        segmentor = None


def image_to_base64(pil_image):
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def parse_via_json(json_file):
    """
    Parse VGG Image Annotator (VIA) 3.x JSON file to count points and extract coordinates.
    Returns: 
        dict: {'count': int, 'points': [[x1, y1], ...]}
    """
    print(f"  [DEBUG] Starting parse_via_json for file: {getattr(json_file, 'filename', 'unknown')}")
    try:
        # Seek to beginning in case it was read before
        json_file.seek(0)
        content = json_file.read()
        print(f"  [DEBUG] Read {len(content)} bytes from JSON file")
        
        if not content:
            print("  [DEBUG] JSON file is empty")
            return {'count': 0, 'points': []}
            
        # Handle bytes vs string
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
            
        data = json.loads(content)
        
        # VIA 3.x stores annotations in the 'metadata' object
        metadata = data.get('metadata', {})
        print(f"  [DEBUG] Found {len(metadata)} metadata entries")
        
        points = []
        for key in metadata:
            region = metadata[key]
            # Coordinates are in 'xy' for VIA 3.x: [type, x, y]
            if 'xy' in region and len(region['xy']) >= 3:
                x, y = region['xy'][1], region['xy'][2]
                points.append([float(x), float(y)])
        
        count = len(points)
        print(f"  [OK] Parsed VIA JSON: {count} points extracted")
        return {'count': count, 'points': points}
    except Exception as e:
        print(f"  [ERROR] Parsing VIA JSON: {e}")
        import traceback
        traceback.print_exc()
        return {'count': 0, 'points': []}


@torch.no_grad()
def run_inference(pil_image):
    """
    Run crowd density estimation on a PIL image.

    Returns:
        dict with predicted count, heatmap base64, overlay base64, etc.
    """
    if model is None:
        raise ValueError("Model not loaded")

    # Prepare image
    img_tensor, original_size = prepare_image_for_inference(pil_image, device)

    # Apply segmentation mask
    mask_b64 = None
    if segmentor is not None:
        img_tensor, mask = segmentor.apply_mask(img_tensor, return_mask=True)
        mask_np = mask.squeeze().cpu().numpy()
        mask_b64 = numpy_to_base64(mask_np, cmap='hot')

    # Model inference
    result = model(img_tensor)
    if isinstance(result, tuple):
        pred_density, aux_logit = result
        scene_prob = torch.sigmoid(aux_logit).item()
        scene_type = "Dense" if scene_prob > 0.5 else "Sparse"
    else:
        pred_density = result
        scene_type = "N/A"
        scene_prob = 0.0

    pred_count = pred_density.sum().item()

    # Generate visualizations
    pred_np = pred_density.squeeze().cpu().numpy()

    # Density heatmap
    heatmap_b64 = numpy_to_base64(pred_np, cmap='jet')

    # Overlay on original image
    img_np = denormalize_image(img_tensor.squeeze(0))
    h_img, w_img = img_np.shape[:2]
    density_up = np.array(
        Image.fromarray(pred_np).resize((w_img, h_img), Image.BILINEAR)
    )
    if density_up.max() > 0:
        density_norm = density_up / density_up.max()
    else:
        density_norm = density_up
    heatmap_rgb = cm.jet(density_norm)[:, :, :3]
    overlay = 0.5 * img_np + 0.5 * heatmap_rgb
    overlay_b64 = numpy_to_base64(overlay, is_overlay=True)


    result = {
        'count': round(pred_count, 1),
        'heatmap': heatmap_b64,
        'overlay': overlay_b64,
        'mask': mask_b64,
        'scene_type': scene_type,
        'scene_confidence': round(scene_prob * 100, 1),
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Optional VIA JSON file
    json_file = request.files.get('json')
    gt_data = {'count': 0, 'points': []}
    has_gt = False
    if json_file and json_file.filename != '':
        print(f"  [DEBUG] JSON file detected: {json_file.filename}")
        gt_data = parse_via_json(json_file)
        has_gt = True
    else:
        print("  [DEBUG] No JSON file provided (Standard Mode)")

    try:
        pil_image = Image.open(image_file.stream).convert('RGB')
        original_b64 = image_to_base64(pil_image)
        print(f"  [DEBUG] Image loaded: {pil_image.size}")

        result = run_inference(pil_image)
        print(f"  [DEBUG] Inference complete: {result['count']} people detected")
        
        result['original'] = original_b64
        result['filename'] = image_file.filename

        # Add comparison metrics if GT is available
        if has_gt:
            gt_count = gt_data['count']
            pred_count = result['count']
            abs_error = abs(pred_count - gt_count)
            result['gt_count'] = gt_count
            result['abs_error'] = round(abs_error, 2)
            result['accuracy'] = round(max(0, 100 - (abs_error / max(gt_count, 1) * 100)), 1)
            print(f"  [DEBUG] Comparison Metrics: GT={gt_count}, Pred={pred_count}, Error={result['abs_error']}")
            
            # Generate GT visualization
            # Need original image in numpy format
            img_tensor, original_size = prepare_image_for_inference(pil_image, device)
            img_np = denormalize_image(img_tensor.squeeze(0))
            
            h_orig, w_orig = original_size[1], original_size[0]
            h_new, w_new = img_np.shape[:2]
            
            from preprocess import generate_density_map_adaptive_fast
            try:
                # Generate density map at original resolution using original points
                gt_points_np = np.array(gt_data['points'])
                gt_density = generate_density_map_adaptive_fast((h_orig, w_orig), gt_points_np)
                
                # Resize density to match new image size for overlay
                density_up = np.array(
                    Image.fromarray(gt_density).resize((w_new, h_new), Image.BILINEAR)
                )
                
                # Normalize and colorize
                if density_up.max() > 0:
                    density_norm = density_up / density_up.max()
                else:
                    density_norm = density_up
                
                heatmap_rgb = cm.jet(density_norm)[:, :, :3]
                
                # Create overlay
                gt_overlay = 0.5 * img_np + 0.5 * heatmap_rgb
                
                from utils import numpy_to_base64, points_to_base64
                result['gt_heatmap'] = numpy_to_base64(density_up, cmap='jet')
                result['gt_overlay'] = numpy_to_base64(gt_overlay, is_overlay=True)
                
                # Generate GT point annotations
                scaled_points = []
                for x, y in gt_data['points']:
                    sx = x * (w_new / w_orig)
                    sy = y * (h_new / h_orig)
                    scaled_points.append([sx, sy])
                result['gt_visualization'] = points_to_base64(img_np, scaled_points, color='yellow')
                
                print(f"  [DEBUG] GT heatmap, overlay, and point visualization generated successfully")
            except Exception as viz_e:
                print(f"  [WARNING] GT visualization failed: {viz_e}")
                import traceback
                traceback.print_exc()
                result['gt_heatmap'] = None
                result['gt_overlay'] = None

        print(f"  [DEBUG] Returning success JSON. Keys: {list(result.keys())}")
        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"  [CRITICAL ERROR] in /predict: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Return info about the loaded model."""
    info = {
        'loaded': model is not None,
        'architecture': model_info.get('architecture', 'N/A'),
        'combined_mae': model_info.get('combined_mae', 'N/A'),
        'dilations': model_info.get('dilations', 'N/A'),
        'segmentation': segmentor is not None,
        'device': str(device),
    }

    # Add per-part metrics if available
    for part in ['A', 'B']:
        key = f'metrics_{part}'
        if key in model_info and model_info[key]:
            m = model_info[key]
            info[f'mae_{part}'] = round(m.get('MAE', 0), 2)
            info[f'rmse_{part}'] = round(m.get('RMSE', 0), 2)

    return jsonify(info)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'segmentation': segmentor is not None,
        'device': str(device),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="CAN+CBAM Crowd Density Web App"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--no_seg", action="store_true",
                        help="Disable segmentation")
    args = parser.parse_args()

    load_model(use_segmentation=not args.no_seg)

    print(f"\n  Starting CAN+CBAM web app on http://localhost:{args.port}")
    print(f"  Model: {'Loaded' if model else 'NOT loaded'}")
    print(f"  Segmentation: {'ON' if segmentor else 'OFF'}")
    app.run(host='0.0.0.0', port=args.port, debug=False)
