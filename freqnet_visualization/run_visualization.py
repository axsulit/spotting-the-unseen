"""
Simple runner script for FreqNet cross-scale consistency visualization.

This script provides an easy way to run the visualization with your specific dataset.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from freqnet_cross_scale_visualizer import FreqNetVisualizer


def run_visualization():
    """Run the visualization with the provided dataset path."""
    
    # Your dataset path
    input_dir = r"D:\ACADEMICS\THESIS\Datasets\final\09_ff40_blur_gaussian_25\train\real"
    
    # Output directory
    output_dir = "./freqnet_visualizations"
    
    # Blur kernel size (matching your dataset)
    blur_kernel_size = 25
    
    # Maximum number of images to process
    max_images = 5
    
    print("=" * 60)
    print("FreqNet Cross-Scale Consistency Visualization")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Blur kernel size: {blur_kernel_size}")
    print(f"Max images to process: {max_images}")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return
    
    # Initialize visualizer
    visualizer = FreqNetVisualizer(device='cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions][:max_images]
    
    if not image_files:
        print(f"ERROR: No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
        try:
            visualizer.visualize_cross_scale_consistency(
                str(image_file), output_dir, blur_kernel_size
            )
            successful += 1
            print(f"✓ Successfully processed {image_file.name}")
        except Exception as e:
            print(f"✗ Error processing {image_file.name}: {e}")
            continue
    
    print(f"\n" + "=" * 60)
    print(f"Visualization complete!")
    print(f"Successfully processed: {successful}/{len(image_files)} images")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    print("\nGenerated visualizations show:")
    print("1. Frequency domain processing steps (HFRI, HFRFC)")
    print("2. Cross-scale consistency breakdown")
    print("3. How blur disrupts multi-resolution dependencies")
    print("4. Channel-wise frequency interactions")
    print("\nThe visualizations demonstrate why FreqNet's performance")
    print("degrades with increasing blur - it breaks the cross-scale")
    print("consistency that the model relies on for accurate classification.")


if __name__ == "__main__":
    run_visualization()