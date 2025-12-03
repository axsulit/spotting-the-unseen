"""
FreqNet Cross-Scale Consistency Visualization

This script visualizes how Gaussian blur affects the cross-scale consistency
in FreqNet's frequency domain processing. It simulates the frequency domain
operations used by FreqNet and shows how blurring disrupts multi-resolution
dependencies and frequency band interactions.

Key FreqNet operations visualized:
1. HFRI (High-Frequency Removal in spatial domain) - hfreqWH()
2. HFRFC (High-Frequency Removal in channel domain) - hfreqC()  
3. FCL (Frequency Convolution Layers) - Complex convolution in frequency domain
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class FreqNetVisualizer:
    """Visualizes FreqNet's frequency domain operations and cross-scale consistency."""
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def hfreqWH(self, x, scale):
        """Apply high-frequency removal in spatial domain via 2D FFT (from FreqNet)."""
        assert scale > 2
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        b, c, h, w = x.shape
        x[:, :, h//2-h//scale:h//2+h//scale, w//2-w//scale:w//2+w//scale] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x
    
    def hfreqC(self, x, scale):
        """Apply high-frequency removal in channel domain via 1D FFT (from FreqNet)."""
        assert scale > 2
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1) 
        b, c, h, w = x.shape
        x[:, c//2-c//scale:c//2+c//scale, :, :] = 0.0
        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x
    
    def get_frequency_spectrum(self, x):
        """Get the magnitude spectrum of input tensor."""
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=[-2, -1])
        magnitude = torch.abs(x_fft)
        return magnitude
    
    def visualize_cross_scale_consistency(self, image_path, output_dir, blur_kernel_size=25):
        """
        Visualize how blur affects cross-scale consistency in FreqNet's frequency processing.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save visualization outputs
            blur_kernel_size: Size of Gaussian blur kernel (5, 10, 15, 20, 25)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Original image tensor
        original_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Apply Gaussian blur
        blurred_image = self.apply_gaussian_blur(image, blur_kernel_size)
        blurred_tensor = transform(blurred_image).unsqueeze(0).to(self.device)
        
        # Get filename for output
        filename = Path(image_path).stem
        
        # Process both original and blurred images through FreqNet operations
        self._process_and_visualize(original_tensor, blurred_tensor, filename, output_dir, blur_kernel_size)
    
    def apply_gaussian_blur(self, image, kernel_size):
        """Apply Gaussian blur to PIL image."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Gaussian blur
        blurred_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        # Convert back to PIL
        return Image.fromarray(blurred_array)
    
    def _process_and_visualize(self, original_tensor, blurred_tensor, filename, output_dir, blur_kernel_size):
        """Process images through FreqNet operations and create visualizations."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process original image
        orig_hfri = self.hfreqWH(original_tensor, 4)
        orig_hfrfc = self.hfreqC(orig_hfri, 4)
        orig_spectrum = self.get_frequency_spectrum(original_tensor)
        
        # Process blurred image
        blur_hfri = self.hfreqWH(blurred_tensor, 4)
        blur_hfrfc = self.hfreqC(blur_hfri, 4)
        blur_spectrum = self.get_frequency_spectrum(blurred_tensor)
        
        # Create comprehensive visualization
        self._create_frequency_visualization(
            original_tensor, blurred_tensor,
            orig_spectrum, blur_spectrum,
            orig_hfri, blur_hfri,
            orig_hfrfc, blur_hfrfc,
            filename, output_dir, blur_kernel_size
        )
        
        # Create cross-scale consistency analysis
        self._create_cross_scale_analysis(
            orig_hfri, blur_hfri,
            orig_hfrfc, blur_hfrfc,
            filename, output_dir, blur_kernel_size
        )
    
    def _create_frequency_visualization(self, orig_tensor, blur_tensor, orig_spectrum, blur_spectrum,
                                      orig_hfri, blur_hfri, orig_hfrfc, blur_hfrfc,
                                      filename, output_dir, blur_kernel_size):
        """Create comprehensive frequency domain visualization."""
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'FreqNet Cross-Scale Consistency Analysis - {filename}\nBlur Kernel Size: {blur_kernel_size}', 
                     fontsize=16, fontweight='bold')
        
        # Row 1: Original Image Processing
        # Original image (denormalized for display)
        orig_display = self._denormalize_tensor(orig_tensor[0])
        axes[0, 0].imshow(orig_display.permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Original frequency spectrum
        orig_spec_display = orig_spectrum[0].mean(dim=0).cpu().numpy()
        im1 = axes[0, 1].imshow(orig_spec_display, cmap='hot', norm=LogNorm())
        axes[0, 1].set_title('Original Frequency Spectrum', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Original after HFRI
        orig_hfri_display = self._denormalize_tensor(orig_hfri[0])
        axes[0, 2].imshow(orig_hfri_display.permute(1, 2, 0).cpu().numpy())
        axes[0, 2].set_title('Original After HFRI (High-Freq Removal)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Original after HFRFC
        orig_hfrfc_display = self._denormalize_tensor(orig_hfrfc[0])
        axes[0, 3].imshow(orig_hfrfc_display.permute(1, 2, 0).cpu().numpy())
        axes[0, 3].set_title('Original After HFRFC (Channel Freq Removal)', fontweight='bold')
        axes[0, 3].axis('off')
        
        # Row 2: Blurred Image Processing
        # Blurred image
        blur_display = self._denormalize_tensor(blur_tensor[0])
        axes[1, 0].imshow(blur_display.permute(1, 2, 0).cpu().numpy())
        axes[1, 0].set_title(f'Blurred Image (Kernel: {blur_kernel_size})', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Blurred frequency spectrum
        blur_spec_display = blur_spectrum[0].mean(dim=0).cpu().numpy()
        im2 = axes[1, 1].imshow(blur_spec_display, cmap='hot', norm=LogNorm())
        axes[1, 1].set_title('Blurred Frequency Spectrum', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Blurred after HFRI
        blur_hfri_display = self._denormalize_tensor(blur_hfri[0])
        axes[1, 2].imshow(blur_hfri_display.permute(1, 2, 0).cpu().numpy())
        axes[1, 2].set_title('Blurred After HFRI', fontweight='bold')
        axes[1, 2].axis('off')
        
        # Blurred after HFRFC
        blur_hfrfc_display = self._denormalize_tensor(blur_hfrfc[0])
        axes[1, 3].imshow(blur_hfrfc_display.permute(1, 2, 0).cpu().numpy())
        axes[1, 3].set_title('Blurred After HFRFC', fontweight='bold')
        axes[1, 3].axis('off')
        
        # Row 3: Cross-Scale Consistency Analysis
        # Difference in HFRI outputs
        hfri_diff = torch.abs(orig_hfri - blur_hfri)
        hfri_diff_display = hfri_diff[0].mean(dim=0).cpu().numpy()
        im3 = axes[2, 0].imshow(hfri_diff_display, cmap='viridis')
        axes[2, 0].set_title('HFRI Difference (Cross-Scale Disruption)', fontweight='bold', color='red')
        axes[2, 0].axis('off')
        plt.colorbar(im3, ax=axes[2, 0], fraction=0.046, pad=0.04)
        
        # Difference in HFRFC outputs
        hfrfc_diff = torch.abs(orig_hfrfc - blur_hfrfc)
        hfrfc_diff_display = hfrfc_diff[0].mean(dim=0).cpu().numpy()
        im4 = axes[2, 1].imshow(hfrfc_diff_display, cmap='viridis')
        axes[2, 1].set_title('HFRFC Difference (Channel Freq Disruption)', fontweight='bold', color='red')
        axes[2, 1].axis('off')
        plt.colorbar(im4, ax=axes[2, 1], fraction=0.046, pad=0.04)
        
        # Frequency spectrum difference
        spec_diff = torch.abs(orig_spectrum - blur_spectrum)
        spec_diff_display = spec_diff[0].mean(dim=0).cpu().numpy()
        im5 = axes[2, 2].imshow(spec_diff_display, cmap='plasma', norm=LogNorm())
        axes[2, 2].set_title('Frequency Spectrum Difference', fontweight='bold', color='red')
        axes[2, 2].axis('off')
        plt.colorbar(im5, ax=axes[2, 2], fraction=0.046, pad=0.04)
        
        # Cross-scale consistency metric
        consistency_score = self._calculate_consistency_score(orig_hfri, blur_hfri, orig_hfrfc, blur_hfrfc)
        axes[2, 3].text(0.5, 0.5, f'Cross-Scale\nConsistency Score:\n{consistency_score:.4f}', 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        axes[2, 3].set_title('Consistency Metric', fontweight='bold', color='red')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}_freqnet_analysis_k{blur_kernel_size}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cross_scale_analysis(self, orig_hfri, blur_hfri, orig_hfrfc, blur_hfrfc,
                                   filename, output_dir, blur_kernel_size):
        """Create detailed cross-scale consistency analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Cross-Scale Consistency Breakdown - {filename}\nBlur Kernel Size: {blur_kernel_size}', 
                     fontsize=16, fontweight='bold')
        
        # Calculate multi-scale differences
        scales = [2, 3, 4, 5, 6]
        consistency_scores = []
        
        for i, scale in enumerate(scales):
            # Apply HFRI at different scales
            orig_scale = self.hfreqWH(orig_hfri, scale)
            blur_scale = self.hfreqWH(blur_hfri, scale)
            
            # Calculate difference
            scale_diff = torch.abs(orig_scale - blur_scale)
            consistency_score = 1.0 - (scale_diff.mean().item() / orig_scale.mean().item())
            consistency_scores.append(consistency_score)
            
            # Visualize difference
            if i < 3:  # Show first 3 scales
                diff_display = scale_diff[0].mean(dim=0).cpu().numpy()
                im = axes[0, i].imshow(diff_display, cmap='Reds')
                axes[0, i].set_title(f'Scale {scale} Difference\nConsistency: {consistency_score:.3f}', 
                                    fontweight='bold')
                axes[0, i].axis('off')
                plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Channel-wise consistency analysis
        channel_diffs = []
        for c in range(orig_hfrfc.shape[1]):
            orig_ch = orig_hfrfc[0, c]
            blur_ch = blur_hfrfc[0, c]
            ch_diff = torch.abs(orig_ch - blur_ch).mean().item()
            channel_diffs.append(ch_diff)
        
        # Plot consistency scores across scales
        axes[1, 0].plot(scales, consistency_scores, 'bo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Frequency Scale')
        axes[1, 0].set_ylabel('Consistency Score')
        axes[1, 0].set_title('Cross-Scale Consistency vs Scale', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Plot channel-wise differences
        axes[1, 1].bar(range(len(channel_diffs)), channel_diffs, color='coral', alpha=0.7)
        axes[1, 1].set_xlabel('Channel Index')
        axes[1, 1].set_ylabel('Mean Absolute Difference')
        axes[1, 1].set_title('Channel-wise Consistency Breakdown', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Overall consistency summary
        overall_score = np.mean(consistency_scores)
        axes[1, 2].text(0.5, 0.7, f'Overall Cross-Scale\nConsistency Score:\n{overall_score:.4f}', 
                        ha='center', va='center', fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].text(0.5, 0.3, f'Blur Kernel Size: {blur_kernel_size}\n\nThis score indicates how\nmuch blur disrupts the\nmulti-resolution dependencies\nthat FreqNet relies on.', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        axes[1, 2].set_title('Summary', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}_cross_scale_analysis_k{blur_kernel_size}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _denormalize_tensor(self, tensor):
        """Denormalize tensor for display."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    def _calculate_consistency_score(self, orig_hfri, blur_hfri, orig_hfrfc, blur_hfrfc):
        """Calculate overall cross-scale consistency score."""
        hfri_diff = torch.abs(orig_hfri - blur_hfri).mean()
        hfrfc_diff = torch.abs(orig_hfrfc - blur_hfrfc).mean()
        
        # Normalize by original values
        hfri_score = 1.0 - (hfri_diff / orig_hfri.mean())
        hfrfc_score = 1.0 - (hfrfc_diff / orig_hfrfc.mean())
        
        return (hfri_score + hfrfc_score).item() / 2.0


def main():
    parser = argparse.ArgumentParser(description='Visualize FreqNet cross-scale consistency breakdown')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing blurred images')
    parser.add_argument('--output_dir', type=str, default='./freqnet_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--blur_kernel', type=int, default=25,
                       help='Blur kernel size (5, 10, 15, 20, 25)')
    parser.add_argument('--max_images', type=int, default=10,
                       help='Maximum number of images to process')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = FreqNetVisualizer(device=args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files
    input_path = Path(args.input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions][:args.max_images]
    
    print(f"Processing {len(image_files)} images from {args.input_dir}")
    print(f"Blur kernel size: {args.blur_kernel}")
    print(f"Output directory: {args.output_dir}")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        try:
            visualizer.visualize_cross_scale_consistency(
                str(image_file), args.output_dir, args.blur_kernel
            )
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue
    
    print(f"\nVisualization complete! Check {args.output_dir} for results.")
    print("\nGenerated visualizations show:")
    print("1. Frequency domain processing steps (HFRI, HFRFC)")
    print("2. Cross-scale consistency breakdown")
    print("3. How blur disrupts multi-resolution dependencies")
    print("4. Channel-wise frequency interactions")


if __name__ == "__main__":
    main()