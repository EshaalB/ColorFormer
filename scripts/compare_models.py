"""
Compare Pretrained vs Our Trained Model
========================================
Generates a side-by-side comparison image showing:
  - Original color image
  - Grayscale input (what the model sees)
  - Pretrained model output (net_g_200000.pth)
  - Our trained model output (net_g_latest.pth)

Usage:
    python scripts/compare_models.py
"""

import argparse
import cv2
import numpy as np
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.colorformer_arch import ColorFormer
from basicsr.utils.img_util import tensor_lab2rgb
from basicsr.data.transforms import rgb2lab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_model(model_path, glh_path='pretrain/GLH.pth',
               color_path='pretrain/color_embed_10000.npy',
               semantic_path='pretrain/semantic_embed_10000.npy',
               input_size=256):
    """Load a ColorFormer model from a checkpoint."""
    model = ColorFormer(
        'GLHTransformer',
        pretrained_path=glh_path,
        input_size=[input_size, input_size],
        num_output_channels=2,
        last_norm='Spectral',
        do_normalize=False,
        color_centers_path=color_path,
        semantic_centers_path=semantic_path
    )
    checkpoint = torch.load(model_path, map_location=device)
    # Handle both formats: with and without 'params' key
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model = model.to(device)
    return model


def colorize_image(model, img_path, input_size=256):
    """Run colorization on a single image and return the result."""
    img = cv2.imread(img_path)
    original = cv2.resize(img, (input_size, input_size))

    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_l, _ = rgb2lab(img_rgb)

    # Prepare input tensors
    img_l_tensor = torch.from_numpy(np.transpose(img_l, (2, 0, 1))).float().unsqueeze(0)
    tensor_lab = torch.cat([img_l_tensor, torch.zeros_like(img_l_tensor), torch.zeros_like(img_l_tensor)], dim=1)
    tensor_rgb = tensor_lab2rgb(tensor_lab)

    # Run inference
    with torch.no_grad():
        imgs = tensor_rgb.to(device)
        img_l_dev = img_l_tensor.to(device)
        outs = model(imgs)
        outs = torch.cat([img_l_dev, outs], dim=1)
        outs = tensor_lab2rgb(outs)

    # Convert output to numpy BGR image
    output = outs[0].cpu().data.float().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output_img = (output * 255.0).round().astype(np.uint8)

    # Make grayscale visualization (what the model sees as input)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return original, gray_bgr, output_img


def add_label(img, text, font_scale=0.6, thickness=2):
    """Add a text label at the top of an image."""
    labeled = img.copy()
    # Add black bar at top
    bar_height = 30
    labeled = np.vstack([np.zeros((bar_height, img.shape[1], 3), dtype=np.uint8), labeled])
    cv2.putText(labeled, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return labeled


def main():
    parser = argparse.ArgumentParser(description='Compare pretrained vs our trained model')
    parser.add_argument('--pretrained_path', type=str, default='pretrain/net_g_200000.pth',
                        help='Path to official pretrained weights')
    parser.add_argument('--our_path', type=str, default='experiments/train_hf/models/net_g_latest.pth',
                        help='Path to our trained weights')
    parser.add_argument('--input', type=str, default='datasets/hf_images',
                        help='Input image folder')
    parser.add_argument('--output', type=str, default='results/comparison',
                        help='Output folder for comparison images')
    parser.add_argument('--input_size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Check if pretrained weights exist
    if not os.path.exists(args.pretrained_path):
        print(f"\n*** ERROR: Pretrained weights not found at: {args.pretrained_path}")
        print("*** Download them from: https://drive.google.com/drive/folders/1ktv0DJFteII4kLb7II0c88jW36aE_hBu")
        print("*** Place net_g_200000.pth in the pretrain/ folder")
        print("\n*** Running with OUR model only for now...\n")
        has_pretrained = False
    else:
        has_pretrained = True

    # Load our trained model
    print("Loading OUR trained model...")
    our_model = load_model(args.our_path)

    # Load pretrained model if available
    if has_pretrained:
        print("Loading PRETRAINED model...")
        pretrained_model = load_model(args.pretrained_path)

    # Get image list
    img_list = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\nProcessing {len(img_list)} images...\n")

    for img_name in img_list:
        img_path = os.path.join(args.input, img_name)

        # Our model
        original, gray, our_result = colorize_image(our_model, img_path, args.input_size)

        if has_pretrained:
            # Pretrained model
            _, _, pretrained_result = colorize_image(pretrained_model, img_path, args.input_size)

            # Build side-by-side: Original | Grayscale | Pretrained | Ours
            labeled_orig = add_label(original, "Original")
            labeled_gray = add_label(gray, "Grayscale Input")
            labeled_pretrained = add_label(pretrained_result, "Pretrained (200k)")
            labeled_ours = add_label(our_result, "Our Fine-tuning (100 iter)")

            comparison = np.hstack([labeled_orig, labeled_gray, labeled_pretrained, labeled_ours])
        else:
            # Build side-by-side: Original | Grayscale | Ours
            labeled_orig = add_label(original, "Original")
            labeled_gray = add_label(gray, "Grayscale Input")
            labeled_ours = add_label(our_result, "Our Fine-tuning (100 iter)")

            comparison = np.hstack([labeled_orig, labeled_gray, labeled_ours])

        output_path = os.path.join(args.output, f"compare_{img_name}")
        cv2.imwrite(output_path, comparison)
        print(f"  Saved: {output_path}")

    print(f"\nDone! {len(img_list)} comparison images saved to: {args.output}/")
    if not has_pretrained:
        print("\nTo get the full 4-panel comparison, download net_g_200000.pth and re-run.")


if __name__ == '__main__':
    main()
