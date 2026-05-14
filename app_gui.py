import gradio as gr
import torch
import cv2
import numpy as np
import os
import sys

# Add current dir to path to import basicsr
sys.path.append(os.getcwd())

from basicsr.archs.colorformer_arch import ColorFormer
from basicsr.utils.img_util import tensor_lab2rgb
from basicsr.data.transforms import rgb2lab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
WEIGHTS_PRETRAIN = 'pretrain/net_g_200000.pth'
WEIGHTS_OURS = 'experiments/train_hf/models/net_g_latest.pth'
WEIGHTS_ENCODER = 'pretrain/GLH.pth'
COLOR_EMBED = 'pretrain/color_embed_10000.npy'
SEMANTIC_EMBED = 'pretrain/semantic_embed_10000.npy'

def load_single_model(weights_path):
    if not os.path.exists(weights_path):
        return None
    try:
        model = ColorFormer(
            'GLHTransformer',
            pretrained_path=WEIGHTS_ENCODER,
            input_size=[256, 256],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            color_centers_path=COLOR_EMBED,
            semantic_centers_path=SEMANTIC_EMBED
        )
        checkpoint = torch.load(weights_path, map_location=device)
        params = checkpoint.get('params', checkpoint)
        model.load_state_dict(params, strict=True)
        model.eval().to(device)
        return model
    except Exception as e:
        print(f"Error loading {weights_path}: {e}")
        return None

# Load both models
model_pretrain = load_single_model(WEIGHTS_PRETRAIN)
model_ours = load_single_model(WEIGHTS_OURS)

status_msg = ""
if model_pretrain: status_msg += "Pretrained Loaded. "
if model_ours: status_msg += "Our Model Loaded."
print(status_msg)

def run_inference(model, input_rgb, img_l_tensor):
    if model is None: return None
    with torch.no_grad():
        out_ab = model(input_rgb)
        img_l_dev = img_l_tensor.to(device)
        out_lab = torch.cat([img_l_dev, out_ab], dim=1) 
        out_rgb = tensor_lab2rgb(out_lab)
    res = (out_rgb[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return res

def process(image):
    # Preprocessing
    h_orig, w_orig = image.shape[:2]
    img_rs = cv2.resize(image, (256, 256))
    
    img_rgb = img_rs.astype(np.float32) / 255.0
    img_l, _ = rgb2lab(img_rgb)
    
    img_l_tensor = torch.from_numpy(np.transpose(img_l, (2, 0, 1))).float().unsqueeze(0)
    tensor_lab_gray = torch.cat([img_l_tensor, torch.zeros_like(img_l_tensor), torch.zeros_like(img_l_tensor)], dim=1)
    input_rgb = tensor_lab2rgb(tensor_lab_gray).to(device)
    
    # Run both
    res_pretrain = run_inference(model_pretrain, input_rgb, img_l_tensor)
    res_ours = run_inference(model_ours, input_rgb, img_l_tensor)
    
    # Resize back
    final_pretrain = cv2.resize(res_pretrain, (w_orig, h_orig)) if res_pretrain is not None else None
    final_ours = cv2.resize(res_ours, (w_orig, h_orig)) if res_ours is not None else None
    
    return final_pretrain, final_ours

# Gradio Interface
with gr.Blocks(title="ColorFormer Comparison GUI") as demo:
    gr.Markdown("# ColorFormer: Pretrained vs. Our Fine-tuning")
    gr.Markdown(f"**Status:** {status_msg}")
    
    with gr.Row():
        input_img = gr.Image(label="Input Gray Image")
    
    with gr.Row():
        out_pre = gr.Image(label="Pretrained (200k iters)")
        out_our = gr.Image(label="Our Fine-tuned (100 iters)")
        
    btn = gr.Button("Colorize & Compare", variant="primary")
    btn.click(fn=process, inputs=input_img, outputs=[out_pre, out_our])

if __name__ == "__main__":
    demo.launch(share=True)
