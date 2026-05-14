import matplotlib.pyplot as plt
import re
import os

log_path = 'experiments/train_hf/train_train_hf_20260512_223853.log'
iters = []
l1_loss = []
percep_loss = []
gan_loss = []

if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        for line in f:
            if 'l_g_pix' in line:
                # Extract iteration
                it_match = re.search(r'iter:\s+(\d+)', line)
                l1_match = re.search(r'l_g_pix:\s+([\d.eE\-\+]+)', line)
                per_match = re.search(r'l_g_percep:\s+([\d.eE\-\+]+)', line)
                gan_match = re.search(r'l_g_gan:\s+([\d.eE\-\+]+)', line)
                
                if it_match and l1_match:
                    iters.append(int(it_match.group(1)))
                    l1_loss.append(float(l1_match.group(1)))
                    if per_match: percep_loss.append(float(per_match.group(1)))
                    if gan_match: gan_loss.append(float(gan_match.group(1)))

    plt.figure(figsize=(10, 6))
    plt.plot(iters, l1_loss, label='Pixel Loss (L1)')
    if percep_loss: plt.plot(iters, percep_loss, label='Perceptual Loss')
    if gan_loss: plt.plot(iters, gan_loss, label='GAN Loss')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('ColorFormer Fine-Tuning Loss Curves')
    plt.legend()
    plt.grid(True)
    
    output_path = 'results/training_loss_plot.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
else:
    print("Log file not found.")
