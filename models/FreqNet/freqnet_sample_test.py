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

from networks.freqnet import freqnet
from data import create_dataloader
from options.test_options import TestOptions

def parse_args():
    parser = argparse.ArgumentParser(description='Test FreqNet model and visualize broken cross-scale consistency')
    parser.add_argument('--model_path', type=str, required=True, help='Path to FreqNet model checkpoint')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to test dataset root directory')
    parser.add_argument('--output_dir', type=str, default='freqnet_samples', help='Directory to save sample images')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to save per category')
    parser.add_argument('--save_frequency_analysis', action='store_true', help='Save frequency domain analysis visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    return parser.parse_args()

def save_image_with_info(image_tensor, true_label, pred_label, category, sample_idx, output_dir, suffix=""):
    """Save image with category information in filename."""
    # Convert tensor to PIL image
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

def visualize_frequency_domain(image_tensor, model, device, output_dir, category, sample_idx, true_label, pred_label):
    """Visualize frequency domain analysis to show cross-scale consistency breakdown."""
    
    model.eval()
    with torch.no_grad():
        # Get intermediate frequency representations
        x = image_tensor.clone()
        
        # Store intermediate representations for visualization
        freq_representations = []
        
        # First HFRI operation
        x_hfri = model.hfreqWH(x, 4)
        freq_representations.append(('After HFRI', x_hfri))
        
        # After first convolution
        x_conv1 = F.conv2d(x_hfri, model.weight1, model.bias1, stride=1, padding=0)
        x_conv1 = F.relu(x_conv1, inplace=True)
        
        # HFRFC operation
        x_hfrfc1 = model.hfreqC(x_conv1, 4)
        freq_representations.append(('After HFRFC1', x_hfrfc1))
        
        # First FCL operation
        x_fcl1 = torch.fft.fft2(x_hfrfc1, norm="ortho")
        x_fcl1 = torch.fft.fftshift(x_fcl1, dim=[-2, -1])
        x_fcl1_complex = torch.complex(model.realconv1(x_fcl1.real), model.imagconv1(x_fcl1.imag))
        x_fcl1 = torch.fft.ifftshift(x_fcl1_complex, dim=[-2, -1])
        x_fcl1 = torch.fft.ifft2(x_fcl1, norm="ortho")
        x_fcl1 = torch.real(x_fcl1)
        x_fcl1 = F.relu(x_fcl1, inplace=True)
        freq_representations.append(('After FCL1', x_fcl1))
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'FreqNet Cross-Scale Analysis - {category} Sample {sample_idx}\nTrue: {true_label}, Pred: {pred_label}', 
                     fontsize=14, fontweight='bold')
        
        # Original image
        orig_denorm = image_tensor.clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        orig_denorm = orig_denorm * std + mean
        orig_denorm = torch.clamp(orig_denorm, 0, 1)
        
        axes[0, 0].imshow(orig_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Frequency domain visualizations
        for i, (name, tensor) in enumerate(freq_representations):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            if row < 2 and col < 3:
                # Compute magnitude spectrum
                freq_mag = torch.abs(torch.fft.fft2(tensor.squeeze(0), norm="ortho"))
                freq_mag = torch.fft.fftshift(freq_mag, dim=[-2, -1])
                
                # Average across channels for visualization
                freq_mag_avg = torch.mean(freq_mag, dim=0)
                
                # Log scale for better visualization
                freq_mag_log = torch.log(freq_mag_avg + 1e-8)
                
                im = axes[row, col].imshow(freq_mag_log.cpu().numpy(), cmap='hot', aspect='auto')
                axes[row, col].set_title(f'{name}\nFrequency Magnitude')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Remove empty subplot if needed
        if len(freq_representations) < 5:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save frequency analysis
        filename = f"{category}_freq_analysis_{sample_idx:02d}_true_{true_label}_pred_{pred_label}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each category
    categories = ['TP', 'TN', 'FP', 'FN']
    for category in categories:
        (output_dir / category).mkdir(exist_ok=True)
    
    # Create frequency analysis subdirectory if requested
    if args.save_frequency_analysis:
        (output_dir / "frequency_analysis").mkdir(exist_ok=True)
    
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
    
    print(f"\nCollecting {args.num_samples} samples from each category to visualize broken cross-scale consistency...")
    print("Categories: TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative)")
    if args.save_frequency_analysis:
        print("Will save frequency domain analysis visualizations")
    
    # Get all dataset indices for random sampling
    dataset_size = len(test_loader.dataset)
    all_indices = list(range(dataset_size))
    
    # Shuffle indices for random sampling
    import random
    random.shuffle(all_indices)
    
    processed_count = 0
    with torch.no_grad():
        for idx in tqdm(all_indices, desc="Testing FreqNet"):
            # Get single sample from dataset
            image, label = test_loader.dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            label = torch.tensor([label]).to(device)
            
            # Run model forward pass
            outputs = model(image)
            
            # Get prediction
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).long()
            
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
            
            # Save images if we haven't reached the limit for this category
            if category_counts[category] < args.num_samples:
                sample_idx = category_counts[category] + 1
                
                # Save original image
                orig_filepath = save_image_with_info(
                    image.squeeze(0), true_label, pred_label, category, sample_idx, 
                    output_dir / category, "_original"
                )
                print(f"Saved {category} original sample {sample_idx}: {orig_filepath.name}")
                
                # Save frequency analysis if requested
                if args.save_frequency_analysis:
                    freq_filepath = visualize_frequency_domain(
                        image.squeeze(0), model, device, output_dir / "frequency_analysis",
                        category, sample_idx, true_label, pred_label
                    )
                    print(f"Saved {category} frequency analysis sample {sample_idx}: {freq_filepath.name}")
                
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
    
    print(f"\nSample images saved to: {output_dir}")
    print("Original faces saved in respective category folders")
    if args.save_frequency_analysis:
        print("Frequency domain analysis saved in 'frequency_analysis' folder")
        print("These visualizations show the breakdown of cross-scale consistency in FreqNet")
    
    print("\nCross-scale consistency analysis complete!")
    print("The frequency visualizations demonstrate how FreqNet's multi-resolution")
    print("dependencies and frequency band interactions are affected by image degradation.")

if __name__ == "__main__":
    main()
