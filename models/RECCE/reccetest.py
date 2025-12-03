import os
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from model.network.Recce import Recce
from dataset import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Test RECCE model and save sample images with reconstructions')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (best_model.pt or latest_model.pt)')
    parser.add_argument('--output_dir', type=str, default='sample_images', help='Directory to save sample images')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to save per category')
    parser.add_argument('--save_reconstructions', action='store_true', help='Save reconstructed faces alongside original faces')
    return parser.parse_args()

def save_image_with_info(image_tensor, true_label, pred_label, category, sample_idx, output_dir, suffix=""):
    """Save image with category information in filename."""
    # Convert tensor to PIL image
    # Denormalize the image
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

def save_side_by_side_comparison(original_tensor, reconstructed_tensor, true_label, pred_label, category, sample_idx, output_dir):
    """Save a side-by-side comparison of original and reconstructed face."""
    # Denormalize both images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_tensor.device)
    
    # Denormalize original
    orig_denorm = original_tensor * std + mean
    orig_denorm = torch.clamp(orig_denorm, 0, 1)
    
    # Denormalize reconstruction (Tanh output is in [-1, 1], convert to [0, 1])
    recon_denorm = (reconstructed_tensor + 1) / 2
    recon_denorm = torch.clamp(recon_denorm, 0, 1)
    
    # Convert to PIL
    orig_pil = transforms.ToPILImage()(orig_denorm)
    recon_pil = transforms.ToPILImage()(recon_denorm)
    
    # Create side-by-side image
    width, height = orig_pil.size
    comparison_img = Image.new('RGB', (width * 2, height))
    comparison_img.paste(orig_pil, (0, 0))
    comparison_img.paste(recon_pil, (width, 0))
    
    # Save comparison image
    filename = f"{category}_comparison_{sample_idx:02d}_true_{true_label}_pred_{pred_label}.png"
    filepath = output_dir / filename
    comparison_img.save(filepath)
    return filepath

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each category
    categories = ['TP', 'TN', 'FP', 'FN']
    for category in categories:
        (output_dir / category).mkdir(exist_ok=True)
    
    # Create comparison subdirectory if saving reconstructions
    if args.save_reconstructions:
        (output_dir / "comparisons").mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(config["train"]["device"])
    print(f"Using device: {device}")
    
    # Initialize model
    model = Recce(num_classes=config["model"]["num_classes"]).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create test dataloader with random sampling
    test_loader = get_dataloader("test")
    
    # Initialize counters for each category
    category_counts = {category: 0 for category in categories}
    
    # Test model
    model.eval()
    
    print(f"\nCollecting {args.num_samples} samples from each category randomly...")
    print("Categories: TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative)")
    if args.save_reconstructions:
        print("Will save original faces, reconstructed faces, and side-by-side comparisons")
    
    # Get all dataset indices for random sampling
    dataset_size = len(test_loader.dataset)
    all_indices = list(range(dataset_size))
    
    # Shuffle indices for random sampling
    import random
    random.shuffle(all_indices)
    
    processed_count = 0
    with torch.no_grad():
        for idx in tqdm(all_indices, desc="Testing"):
            # Get single sample from dataset
            image, label = test_loader.dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            label = torch.tensor([label]).to(device)
            
            # Run model forward pass
            outputs = model(image)
            
            # Extract reconstruction from loss_inputs
            reconstructed_face = None
            if args.save_reconstructions and 'recons' in model.loss_inputs and model.loss_inputs['recons']:
                reconstructed_face = model.loss_inputs['recons'][0].squeeze(0)  # Remove batch dimension
            
            # Ensure outputs is 2D
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)  # Add batch dimension
            elif outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)  # Add batch dimension if missing
            
            outputs = outputs.squeeze()
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
                
                # Save reconstructed image if requested
                if args.save_reconstructions and reconstructed_face is not None:
                    recon_filepath = save_image_with_info(
                        reconstructed_face, true_label, pred_label, category, sample_idx,
                        output_dir / category, "_reconstructed"
                    )
                    print(f"Saved {category} reconstructed sample {sample_idx}: {recon_filepath.name}")
                    
                    # Save side-by-side comparison
                    comparison_filepath = save_side_by_side_comparison(
                        image.squeeze(0), reconstructed_face, true_label, pred_label,
                        category, sample_idx, output_dir / "comparisons"
                    )
                    print(f"Saved {category} comparison sample {sample_idx}: {comparison_filepath.name}")
                
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
    if args.save_reconstructions:
        print("Original faces saved with '_original' suffix")
        print("Reconstructed faces saved with '_reconstructed' suffix") 
        print("Side-by-side comparisons saved in 'comparisons' folder")
    print("Exiting early - no metrics calculated as requested.")

if __name__ == "__main__":
    main()
