import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import random

from networks.freqnet import freqnet
from data import create_dataloader
from options.test_options import TestOptions

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FreqNet cross-scale consistency breakdown')
    parser.add_argument('--model_path', type=str, required=True, help='Path to FreqNet model checkpoint')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to test dataset root directory')
    parser.add_argument('--output_dir', type=str, default='freqnet_crossscale_analysis', help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples per category (TP, TN, FP, FN)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    return parser.parse_args()

def save_image_with_info(image_tensor, true_label, pred_label, category, sample_idx, output_dir, suffix=""):
    """Save image with category information in filename."""
    # Denormalize the image (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    image_tensor = image_tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Convert to PIL
    image = transforms.ToPILImage()(image_tensor)
    
    # Create filename with category info
    filename = f"{category}_sample_{sample_idx:02d}_true_{true_label}_pred_{pred_label}{suffix}.png"
    filepath = output_dir / filename
    image.save(filepath)
    return filepath

def visualize_hfri_output(image_tensor, model, device):
    """Visualize HFRI (High-Frequency Removal) block output with FFT magnitude spectrum."""
    with torch.no_grad():
        # Ensure image_tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Apply HFRI operation (scale=4 as used in FreqNet)
        hfri_output = model.hfreqWH(image_tensor, 4)
        
        # Compute FFT magnitude spectrum
        fft_mag = torch.abs(torch.fft.fft2(hfri_output.squeeze(0), norm="ortho"))
        fft_mag = torch.fft.fftshift(fft_mag, dim=[-2, -1])
        
        # Average across channels for visualization
        fft_mag_avg = torch.mean(fft_mag, dim=0)
        
        return hfri_output, fft_mag_avg

def visualize_hfrf_output(image_tensor, model, device):
    """Visualize HFRF (High-Frequency Removal Features) at multiple scales."""
    with torch.no_grad():
        # Ensure image_tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Get intermediate outputs at different scales
        # Scale 1: After first HFRI + Conv
        x1 = model.hfreqWH(image_tensor, 4)
        x1 = F.conv2d(x1, model.weight1, model.bias1, stride=1, padding=0)
        x1 = F.relu(x1, inplace=True)
        
        # Scale 2: After HFRFC + FCL
        x2 = model.hfreqC(x1, 4)
        x2_fft = torch.fft.fft2(x2, norm="ortho")
        x2_fft = torch.fft.fftshift(x2_fft, dim=[-2, -1])
        x2_complex = torch.complex(model.realconv1(x2_fft.real), model.imagconv1(x2_fft.imag))
        x2_fft = torch.fft.ifftshift(x2_complex, dim=[-2, -1])
        x2 = torch.fft.ifft2(x2_fft, norm="ortho")
        x2 = torch.real(x2)
        x2 = F.relu(x2, inplace=True)
        
        # Scale 3: After second HFRI + Conv (stride=2)
        x3 = model.hfreqWH(x2, 4)
        x3 = F.conv2d(x3, model.weight2, model.bias2, stride=2, padding=0)
        x3 = F.relu(x3, inplace=True)
        
        # Select representative channels for visualization
        channels_to_show = [0, 15, 31]  # Show channels 0, 15, 31
        
        return {
            'scale1': x1.squeeze(0)[channels_to_show],  # [3, H, W]
            'scale2': x2.squeeze(0)[channels_to_show],  # [3, H, W]
            'scale3': x3.squeeze(0)[channels_to_show]   # [3, H, W]
        }

def compute_cross_scale_correlation(features_dict):
    """Compute cross-scale correlation heatmap."""
    scales = ['scale1', 'scale2', 'scale3']
    correlation_matrix = np.zeros((len(scales), len(scales)))
    
    # Get the smallest spatial dimensions to resize all scales to the same size
    min_h, min_w = float('inf'), float('inf')
    for scale_name, features in features_dict.items():
        h, w = features.shape[1], features.shape[2]
        min_h, min_w = min(min_h, h), min(min_w, w)
    
    for i, scale1 in enumerate(scales):
        for j, scale2 in enumerate(scales):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # Resize both features to the same spatial dimensions
                feat1 = F.interpolate(features_dict[scale1].unsqueeze(0), size=(min_h, min_w), mode='bilinear', align_corners=False).squeeze(0)
                feat2 = F.interpolate(features_dict[scale2].unsqueeze(0), size=(min_h, min_w), mode='bilinear', align_corners=False).squeeze(0)
                
                # Flatten features for correlation computation
                feat1_flat = feat1.flatten().cpu().numpy()
                feat2_flat = feat2.flatten().cpu().numpy()
                
                # Compute Pearson correlation
                corr, _ = pearsonr(feat1_flat, feat2_flat)
                correlation_matrix[i, j] = corr
    
    return correlation_matrix

def visualize_fcl_output(image_tensor, model, device):
    """Visualize FCL (Frequency Convolution Layer) outputs showing amplitude/phase spectrum."""
    with torch.no_grad():
        # Ensure image_tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Get intermediate features before FCL
        x = model.hfreqWH(image_tensor, 4)
        x = F.conv2d(x, model.weight1, model.bias1, stride=1, padding=0)
        x = F.relu(x, inplace=True)
        x = model.hfreqC(x, 4)
        
        # Apply FCL operation
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=[-2, -1])
        
        # Apply learned convolutions
        x_real = model.realconv1(x_fft.real)
        x_imag = model.imagconv1(x_fft.imag)
        x_complex = torch.complex(x_real, x_imag)
        
        # Convert back to spatial domain
        x_fft = torch.fft.ifftshift(x_complex, dim=[-2, -1])
        x_out = torch.fft.ifft2(x_fft, norm="ortho")
        x_out = torch.real(x_out)
        x_out = F.relu(x_out, inplace=True)
        
        # Extract amplitude and phase
        amplitude = torch.abs(x_complex)
        phase = torch.angle(x_complex)
        
        # Average across channels for visualization
        amp_avg = torch.mean(amplitude, dim=1).squeeze(0)  # [H, W]
        phase_avg = torch.mean(phase, dim=1).squeeze(0)   # [H, W]
        
        return {
            'amplitude': amp_avg,
            'phase': phase_avg,
            'output': x_out.squeeze(0)
        }

def create_comprehensive_visualization(image_tensor, model, device, category, sample_idx, true_label, pred_label, confidence, output_dir):
    """Create comprehensive multi-panel visualization of FreqNet cross-scale consistency."""
    
    # Ensure image_tensor has batch dimension for model operations
    if image_tensor.dim() == 3:
        image_tensor_batch = image_tensor.unsqueeze(0)
    else:
        image_tensor_batch = image_tensor
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Set up the grid layout
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'FreqNet Cross-Scale Consistency Analysis - {category} Sample {sample_idx}\n'
                f'True Label: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.3f}', 
                fontsize=16, fontweight='bold')
    
    # 1. Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    orig_denorm = image_tensor.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    orig_denorm = orig_denorm * std + mean
    orig_denorm = torch.clamp(orig_denorm, 0, 1)
    
    ax1.imshow(orig_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Input Image\n(May be blurred)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. HFRI Block Output
    ax2 = fig.add_subplot(gs[0, 1])
    hfri_output, fft_mag = visualize_hfri_output(image_tensor_batch, model, device)
    
    # Show HFRI output (spatial domain)
    hfri_vis = torch.mean(hfri_output.squeeze(0), dim=0)
    im2 = ax2.imshow(hfri_vis.cpu().numpy(), cmap='viridis')
    ax2.set_title('HFRI Block Output\n(Spatial Domain)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. HFRI FFT Magnitude Spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(torch.log(fft_mag + 1e-8).cpu().numpy(), cmap='hot', aspect='auto')
    ax3.set_title('HFRI FFT Magnitude\nSpectrum (Log Scale)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4-6. HFRF Multi-Scale Features
    hfrf_features = visualize_hfrf_output(image_tensor_batch, model, device)
    
    # Only show first 2 scales to fit in the grid (columns 3 and 4)
    scale_items = list(hfrf_features.items())[:2]  # Take only first 2 scales
    for i, (scale_name, features) in enumerate(scale_items):
        ax = fig.add_subplot(gs[0, 3+i])
        
        # Show average of selected channels
        avg_features = torch.mean(features, dim=0)
        im = ax.imshow(avg_features.cpu().numpy(), cmap='plasma')
        ax.set_title(f'HFRF {scale_name.upper()}\nFeature Maps', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 7. Cross-Scale Correlation Heatmap
    ax7 = fig.add_subplot(gs[1, :2])
    correlation_matrix = compute_cross_scale_correlation(hfrf_features)
    
    im7 = ax7.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax7.set_title('Cross-Scale Correlation\nHeatmap', fontsize=12, fontweight='bold')
    ax7.set_xticks([0, 1, 2])
    ax7.set_yticks([0, 1, 2])
    ax7.set_xticklabels(['Scale1', 'Scale2', 'Scale3'])
    ax7.set_yticklabels(['Scale1', 'Scale2', 'Scale3'])
    
    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            text = ax7.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # 8-10. FCL Output Analysis
    fcl_output = visualize_fcl_output(image_tensor_batch, model, device)
    
    # FCL Amplitude Spectrum
    ax8 = fig.add_subplot(gs[1, 2])
    im8 = ax8.imshow(fcl_output['amplitude'].cpu().numpy(), cmap='hot', aspect='auto')
    ax8.set_title('FCL Amplitude\nSpectrum', fontsize=12, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # FCL Phase Spectrum
    ax9 = fig.add_subplot(gs[1, 3])
    im9 = ax9.imshow(fcl_output['phase'].cpu().numpy(), cmap='hsv', aspect='auto')
    ax9.set_title('FCL Phase\nSpectrum', fontsize=12, fontweight='bold')
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
    
    # FCL Output (Spatial Domain)
    ax10 = fig.add_subplot(gs[1, 4])
    fcl_spatial = torch.mean(fcl_output['output'], dim=0)
    im10 = ax10.imshow(fcl_spatial.cpu().numpy(), cmap='viridis')
    ax10.set_title('FCL Output\n(Spatial Domain)', fontsize=12, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)
    
    # 11-13. Individual Channel Visualizations (Scale 1)
    ax11 = fig.add_subplot(gs[2, 0])
    im11 = ax11.imshow(hfrf_features['scale1'][0].cpu().numpy(), cmap='plasma')
    ax11.set_title('Scale1 - Channel 0', fontsize=10)
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)
    
    ax12 = fig.add_subplot(gs[2, 1])
    im12 = ax12.imshow(hfrf_features['scale1'][1].cpu().numpy(), cmap='plasma')
    ax12.set_title('Scale1 - Channel 15', fontsize=10)
    ax12.axis('off')
    plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)
    
    ax13 = fig.add_subplot(gs[2, 2])
    im13 = ax13.imshow(hfrf_features['scale1'][2].cpu().numpy(), cmap='plasma')
    ax13.set_title('Scale1 - Channel 31', fontsize=10)
    ax13.axis('off')
    plt.colorbar(im13, ax=ax13, fraction=0.046, pad=0.04)
    
    # 14-16. Individual Channel Visualizations (Scale 2)
    ax14 = fig.add_subplot(gs[2, 3])
    im14 = ax14.imshow(hfrf_features['scale2'][0].cpu().numpy(), cmap='plasma')
    ax14.set_title('Scale2 - Channel 0', fontsize=10)
    ax14.axis('off')
    plt.colorbar(im14, ax=ax14, fraction=0.046, pad=0.04)
    
    ax15 = fig.add_subplot(gs[2, 4])
    im15 = ax15.imshow(hfrf_features['scale2'][1].cpu().numpy(), cmap='plasma')
    ax15.set_title('Scale2 - Channel 15', fontsize=10)
    ax15.axis('off')
    plt.colorbar(im15, ax=ax15, fraction=0.046, pad=0.04)
    
    # 17-19. Individual Channel Visualizations (Scale 3) - Show in row 3
    ax17 = fig.add_subplot(gs[3, 0])
    im17 = ax17.imshow(hfrf_features['scale3'][0].cpu().numpy(), cmap='plasma')
    ax17.set_title('Scale3 - Channel 0', fontsize=10)
    ax17.axis('off')
    plt.colorbar(im17, ax=ax17, fraction=0.046, pad=0.04)
    
    ax18 = fig.add_subplot(gs[3, 1])
    im18 = ax18.imshow(hfrf_features['scale3'][1].cpu().numpy(), cmap='plasma')
    ax18.set_title('Scale3 - Channel 15', fontsize=10)
    ax18.axis('off')
    plt.colorbar(im18, ax=ax18, fraction=0.046, pad=0.04)
    
    ax19 = fig.add_subplot(gs[3, 2])
    im19 = ax19.imshow(hfrf_features['scale3'][2].cpu().numpy(), cmap='plasma')
    ax19.set_title('Scale3 - Channel 31', fontsize=10)
    ax19.axis('off')
    plt.colorbar(im19, ax=ax19, fraction=0.046, pad=0.04)
    
    # 20. Summary Statistics
    ax20 = fig.add_subplot(gs[3, 3:])
    ax20.axis('off')
    
    # Compute summary statistics
    avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
    min_correlation = np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
    max_correlation = np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
    
    summary_text = f"""
Cross-Scale Consistency Analysis Summary:

• Average Cross-Scale Correlation: {avg_correlation:.3f}
• Min Cross-Scale Correlation: {min_correlation:.3f}
• Max Cross-Scale Correlation: {max_correlation:.3f}

Interpretation:
• High correlations (>0.7): Strong cross-scale consistency
• Medium correlations (0.3-0.7): Moderate consistency
• Low correlations (<0.3): Broken cross-scale consistency

Category: {category}
• TP: Correctly identified fake
• TN: Correctly identified real  
• FP: Incorrectly identified as fake
• FN: Incorrectly identified as real
"""
    
    ax20.text(0.05, 0.95, summary_text, transform=ax20.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Save the comprehensive visualization
    filename = f"{category}_crossscale_analysis_{sample_idx:02d}_true_{true_label}_pred_{pred_label}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath

def main():
    args = parse_args()
    
    # Set random seed for reproducible sampling
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each category
    categories = ['TP', 'TN', 'FP', 'FN']
    for category in categories:
        (output_dir / category).mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = freqnet(num_classes=1, device=device).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded FreqNet model from: {args.model_path}")
    
    # Create test options
    test_opt = TestOptions().parse(print_options=False)
    test_opt.dataroot = args.dataroot
    test_opt.classes = ['real', 'fake']
    test_opt.batch_size = 1  # Process one image at a time
    test_opt.num_threads = 1
    
    # Create test dataloader
    test_loader = create_dataloader(test_opt)
    
    # Initialize counters for each category
    category_counts = {category: 0 for category in categories}
    
    # Test model
    model.eval()
    
    print(f"\nVisualizing FreqNet cross-scale consistency breakdown...")
    print(f"Collecting {args.num_samples} samples from each category")
    print("Categories: TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative)")
    print("This will show how blur disrupts frequency-based feature consistency in FreqNet")
    
    # Get all dataset indices for random sampling
    dataset_size = len(test_loader.dataset)
    all_indices = list(range(dataset_size))
    
    # Shuffle indices for random sampling
    random.shuffle(all_indices)
    
    processed_count = 0
    with torch.no_grad():
        for idx in tqdm(all_indices, desc="Analyzing FreqNet"):
            # Get single sample from dataset
            image, label = test_loader.dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            label = torch.tensor([label]).to(device)
            
            # Run model forward pass
            outputs = model(image)
            
            # Get prediction and confidence
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).long()
            confidence = probs.item()
            
            true_label = label.item()
            pred_label = predictions.item()
            
            # Determine category
            if true_label == 1 and pred_label == 1:
                category = 'TP'
            elif true_label == 0 and pred_label == 0:
                category = 'TN'
            elif true_label == 0 and pred_label == 1:
                category = 'FP'
            elif true_label == 1 and pred_label == 0:
                category = 'FN'
            else:
                continue  # Shouldn't happen, but just in case
            
            # Create visualizations if we haven't reached the limit for this category
            if category_counts[category] < args.num_samples:
                sample_idx = category_counts[category] + 1
                
                # Save original image
                orig_filepath = save_image_with_info(
                    image.squeeze(0), true_label, pred_label, category, sample_idx, 
                    output_dir / category, "_original"
                )
                print(f"Saved {category} original sample {sample_idx}: {orig_filepath.name}")
                
                # Create comprehensive cross-scale consistency visualization
                viz_filepath = create_comprehensive_visualization(
                    image.squeeze(0), model, device, category, sample_idx, 
                    true_label, pred_label, confidence, output_dir / category
                )
                print(f"Created {category} cross-scale analysis sample {sample_idx}: {viz_filepath.name}")
                
                category_counts[category] += 1
            
            processed_count += 1
            
            # Check if we have enough samples from all categories
            if all(count >= args.num_samples for count in category_counts.values()):
                print(f"\nCollected {args.num_samples} samples from all categories!")
                print(f"Processed {processed_count} samples out of {dataset_size} total samples")
                break
    
    # Print final summary
    print(f"\nFinal counts:")
    for category, count in category_counts.items():
        print(f"{category}: {count}/{args.num_samples}")
    
    print(f"\nCross-scale consistency analysis complete!")
    print(f"Visualizations saved to: {output_dir}")
    print("\nKey findings:")
    print("• Original images should show strong cross-scale correlations")
    print("• Blurred images should show collapsed correlations (broken consistency)")
    print("• This visualizes why FreqNet performance drops under Gaussian blur")
    print("• The multi-panel figures show HFRI, HFRF, and FCL block outputs")
    print("• Correlation heatmaps quantify cross-scale consistency breakdown")

if __name__ == "__main__":
    main()
