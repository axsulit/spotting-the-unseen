"""
FreqNet HFRI/HFRF Block Visualization Script

This script visualizes how different datasets (FF++ c23, FF++ c40, Celeb-DF, WDF)
appear in the model's HFRI and HFRF blocks, comparing Real vs Fake images.

Outputs:
- Per-dataset HFR visualizations
- Summary comparison across datasets
- Statistics CSV
- Raw numpy arrays for further analysis
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# ============================================================================
# PROCESSING PARAMETERS (adjustable at the top)
# ============================================================================
num_samples_per_class = 50    # number of images to sample (per real/fake) for averaging
image_size = (256, 256)       # resize images for network / FFT
layer_probe_candidates = ["HFRI", "HFRF", "hfr", "freq", "freq_conv", "freq_layer"]
feature_layer_fallback = ["features.5", "layer3", "layer2", "conv3", "conv4"]  # try these in order if HFR names absent
center_frac = 0.5              # central fraction to zero-out when computing HF (optional)

# ============================================================================
# PLACEHOLDER PATHS (update with actual paths)
# ============================================================================
CHECKPOINT_PATHS = {
    "FFpp_c23": "path/to/freqnet_ffpp_c23_checkpoint.pth",
    "FFpp_c40": "path/to/freqnet_ffpp_c40_checkpoint.pth",
    "CelebDF": "path/to/freqnet_celebdF_checkpoint.pth",
    "WDF": "path/to/freqnet_wdf_checkpoint.pth"
}

DATASET_DIRS = {
    "FFpp_c23": "path/to/FFpp_c23_dataset",  # should contain real/ and fake/ subfolders
    "FFpp_c40": "path/to/FFpp_c40_dataset",
    "CelebDF": "path/to/CelebDF_dataset",
    "WDF": "path/to/WDF_dataset"
}

OUTPUT_DIR = "freqnet_hfr_vis"

# ============================================================================
# SETUP
# ============================================================================
def setup_output_dir(output_dir):
    """Create output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "raw_arrays").mkdir(exist_ok=True)
    return output_path

# ============================================================================
# MODEL LOADING & PROBE DISCOVERY
# ============================================================================
def load_checkpoint(checkpoint_path, device):
    """Load checkpoint, handling different formats."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            else:
                return checkpoint
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def find_hfr_layers(model):
    """Search for HFRI/HFRF layers in the model."""
    hfr_layers = {}
    layer_names = []
    
    def get_layer_names(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layer_names.append((full_name, child))
            get_layer_names(child, full_name)
    
    get_layer_names(model)
    
    # Search for HFR-related layers
    for name, layer in layer_names:
        name_lower = name.lower()
        for candidate in layer_probe_candidates:
            if candidate.lower() in name_lower:
                if 'hfri' in name_lower or 'hfri' in candidate.lower():
                    hfr_layers['HFRI'] = (name, layer)
                elif 'hfrf' in name_lower or 'hfrf' in candidate.lower():
                    hfr_layers['HFRF'] = (name, layer)
                else:
                    # Generic HFR layer
                    if 'HFRI' not in hfr_layers:
                        hfr_layers['HFRI'] = (name, layer)
                    if 'HFRF' not in hfr_layers and name != hfr_layers.get('HFRI', ('', None))[0]:
                        hfr_layers['HFRF'] = (name, layer)
    
    return hfr_layers

def find_fallback_layer(model, fallback_names):
    """Find a fallback convolutional layer if HFR layers not found."""
    for name in fallback_names:
        parts = name.split('.')
        layer = model
        try:
            for part in parts:
                layer = getattr(layer, part)
            if isinstance(layer, (nn.Conv2d, nn.Sequential)):
                return name, layer
        except AttributeError:
            continue
    return None, None

def register_hooks(model, device):
    """Register forward hooks to capture HFRI/HFRF activations."""
    activations = {}
    hooks = []
    hook_handles = []
    
    # Store original methods for cleanup
    original_hfreqWH = model.hfreqWH if hasattr(model, 'hfreqWH') else None
    original_hfreqC = model.hfreqC if hasattr(model, 'hfreqC') else None
    
    # Create wrapper methods that capture outputs
    # We'll capture feature maps at key points in the forward pass
    def hfreqWH_wrapper(x, scale):
        output = original_hfreqWH(x, scale)
        # Store the output as a feature map for HFRI computation
        # This is after high-frequency removal, so we'll compute FFT on this
        activations['HFRI_feature'] = output.detach().cpu()
        return output
    
    def hfreqC_wrapper(x, scale):
        output = original_hfreqC(x, scale)
        # Store the output as a feature map for HFRF computation
        activations['HFRF_feature'] = output.detach().cpu()
        return output
    
    # Patch methods if they exist
    if original_hfreqWH is not None:
        model.hfreqWH = hfreqWH_wrapper
        activations['HFRI_feature'] = None
    
    if original_hfreqC is not None:
        model.hfreqC = hfreqC_wrapper
        activations['HFRF_feature'] = None
    
    # Also try to register hooks on conv layers as fallback
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
        return hook
    
    # Try to find fallback layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and any(fallback in name for fallback in feature_layer_fallback):
            handle = module.register_forward_hook(make_hook(f'fallback_{name}'))
            hook_handles.append(handle)
            activations[f'fallback_{name}'] = None
            break  # Use first matching layer
    
    # Store cleanup info
    hooks.append(('hfreqWH', original_hfreqWH))
    hooks.append(('hfreqC', original_hfreqC))
    hooks.append(('hook_handles', hook_handles))
    
    return activations, hooks

def load_model(checkpoint_path, device):
    """Load FreqNet model from checkpoint."""
    # Try to import FreqNet
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from networks.freqnet import freqnet
        model = freqnet(num_classes=1, device=device).to(device)
        model.eval()
        
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint is not None:
            try:
                model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded model from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load state dict: {e}")
                print("Proceeding with image-level FFT calculations")
                return None, None, None
    except ImportError:
        print("Warning: Could not import FreqNet. Proceeding with image-level FFT.")
        return None, None, None
    
    # Register hooks
    activations, hooks = register_hooks(model, device)
    
    return model, activations, hooks

# ============================================================================
# DATA SAMPLING & PREPROCESSING
# ============================================================================
def get_image_files(dataset_dir, class_name, num_samples, seed=42):
    """Get image files from dataset directory."""
    class_dir = Path(dataset_dir) / class_name
    if not class_dir.exists():
        print(f"Warning: {class_dir} does not exist")
        return []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_files = [f for f in class_dir.rglob('*') if f.suffix.lower() in image_extensions]
    
    if len(all_files) == 0:
        print(f"Warning: No images found in {class_dir}")
        return []
    
    # Random sample with deterministic seed
    random.seed(seed)
    np.random.seed(seed)
    sampled = random.sample(all_files, min(num_samples, len(all_files)))
    
    if len(sampled) < num_samples:
        print(f"Warning: Only {len(sampled)} images available in {class_dir}, requested {num_samples}")
    
    return sorted(sampled)  # Sort for reproducibility

def preprocess_image(image_path, image_size):
    """Load and preprocess image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def get_representative_samples(dataset_dir, class_name, num_samples=3, seed=42):
    """Get a small set of representative images for visualization."""
    files = get_image_files(dataset_dir, class_name, num_samples, seed)
    samples = []
    for f in files:
        img = preprocess_image(f, image_size)
        if img is not None:
            samples.append(img)
    return samples

# ============================================================================
# ACTIVATION EXTRACTION & HFR COMPUTATION
# ============================================================================
def compute_fft_hfr(image_array, center_frac=0.5):
    """Compute HFRF (amplitude) and HFRI (phase) from image using FFT."""
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array
    
    # Compute FFT
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute amplitude and phase
    amplitude = np.log1p(np.abs(fft_shifted))
    phase = np.angle(fft_shifted)
    
    # Compute high-frequency energy
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    mask_h = int(h * center_frac / 2)
    mask_w = int(w * center_frac / 2)
    
    # Create mask for high frequencies (outside center)
    mask = np.ones((h, w), dtype=bool)
    mask[center_h - mask_h:center_h + mask_h, center_w - mask_w:center_w + mask_w] = False
    hf_energy = np.sum(amplitude[mask])
    
    return amplitude, phase, hf_energy

def compute_hfr_from_activation(activation_tensor, center_frac=0.5):
    """Compute HFRF/HFRI from model activation tensor."""
    # Handle batch dimension
    if activation_tensor.dim() == 4:
        activation = activation_tensor[0]  # Take first batch item
    else:
        activation = activation_tensor
    
    # activation shape: C x H x W
    C, H, W = activation.shape
    
    # Compute FFT for each channel
    amplitudes = []
    phases = []
    
    for c in range(C):
        channel_data = activation[c].numpy()
        fft = np.fft.fft2(channel_data)
        fft_shifted = np.fft.fftshift(fft)
        
        amp = np.log1p(np.abs(fft_shifted))
        ph = np.angle(fft_shifted)
        
        amplitudes.append(amp)
        phases.append(ph)
    
    # Average across channels
    amp_avg = np.mean(amplitudes, axis=0)
    phase_avg = np.mean(phases, axis=0)
    
    # Compute high-frequency energy
    center_h, center_w = H // 2, W // 2
    mask_h = int(H * center_frac / 2)
    mask_w = int(W * center_frac / 2)
    
    mask = np.ones((H, W), dtype=bool)
    mask[center_h - mask_h:center_h + mask_h, center_w - mask_w:center_w + mask_w] = False
    hf_energy = np.sum(amp_avg[mask])
    
    return amp_avg, phase_avg, hf_energy

def process_dataset(dataset_name, dataset_dir, checkpoint_path, device, output_dir):
    """Process a single dataset."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Check dataset structure
    real_dir = Path(dataset_dir) / "real"
    fake_dir = Path(dataset_dir) / "fake"
    
    if not real_dir.exists() or not fake_dir.exists():
        print(f"Error: Dataset structure invalid. Expected real/ and fake/ subfolders in {dataset_dir}")
        return None
    
    # Load model
    model, activations, hooks = load_model(checkpoint_path, device)
    use_model = model is not None
    
    # Get image files
    real_files = get_image_files(dataset_dir, "real", num_samples_per_class, seed=42)
    fake_files = get_image_files(dataset_dir, "fake", num_samples_per_class, seed=42)
    
    print(f"Found {len(real_files)} real images, {len(fake_files)} fake images")
    
    # Process real images
    print("Processing real images...")
    real_amps = []
    real_phases = []
    real_hf_energies = []
    
    for img_path in tqdm(real_files, desc="Real"):
        img_array = preprocess_image(img_path, image_size)
        if img_array is None:
            continue
        
        if use_model:
            # Try to get model activations
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
            # Normalize if needed (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            with torch.no_grad():
                # Forward pass to get activations
                # Clear previous activations
                if activations:
                    for key in activations:
                        activations[key] = None
                
                _ = model(img_tensor)
                
                # Try to extract from activations dict
                # Prefer HFRF_feature or HFRI_feature, then fallback layers
                feature_map = None
                if activations and 'HFRF_feature' in activations and activations['HFRF_feature'] is not None:
                    feature_map = activations['HFRF_feature']
                elif activations and 'HFRI_feature' in activations and activations['HFRI_feature'] is not None:
                    feature_map = activations['HFRI_feature']
                else:
                    # Try fallback layers
                    for key in activations:
                        if key.startswith('fallback_') and activations[key] is not None:
                            feature_map = activations[key]
                            break
                
                if feature_map is not None:
                    amp, phase, hf_energy = compute_hfr_from_activation(feature_map, center_frac)
                else:
                    # Fallback to image-level FFT
                    amp, phase, hf_energy = compute_fft_hfr(img_array, center_frac)
        else:
            # Image-level FFT
            amp, phase, hf_energy = compute_fft_hfr(img_array, center_frac)
        
        real_amps.append(amp)
        real_phases.append(phase)
        real_hf_energies.append(hf_energy)
    
    # Process fake images
    print("Processing fake images...")
    fake_amps = []
    fake_phases = []
    fake_hf_energies = []
    
    for img_path in tqdm(fake_files, desc="Fake"):
        img_array = preprocess_image(img_path, image_size)
        if img_array is None:
            continue
        
        if use_model:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            with torch.no_grad():
                # Clear previous activations
                if activations:
                    for key in activations:
                        activations[key] = None
                
                _ = model(img_tensor)
                
                # Try to extract from activations dict
                feature_map = None
                if activations and 'HFRF_feature' in activations and activations['HFRF_feature'] is not None:
                    feature_map = activations['HFRF_feature']
                elif activations and 'HFRI_feature' in activations and activations['HFRI_feature'] is not None:
                    feature_map = activations['HFRI_feature']
                else:
                    # Try fallback layers
                    for key in activations:
                        if key.startswith('fallback_') and activations[key] is not None:
                            feature_map = activations[key]
                            break
                
                if feature_map is not None:
                    amp, phase, hf_energy = compute_hfr_from_activation(feature_map, center_frac)
                else:
                    amp, phase, hf_energy = compute_fft_hfr(img_array, center_frac)
        else:
            amp, phase, hf_energy = compute_fft_hfr(img_array, center_frac)
        
        fake_amps.append(amp)
        fake_phases.append(phase)
        fake_hf_energies.append(hf_energy)
    
    # Aggregate
    real_amp_mean = np.mean(real_amps, axis=0) if real_amps else None
    real_phase_mean = np.mean(real_phases, axis=0) if real_phases else None
    fake_amp_mean = np.mean(fake_amps, axis=0) if fake_amps else None
    fake_phase_mean = np.mean(fake_phases, axis=0) if fake_phases else None
    
    amp_diff = fake_amp_mean - real_amp_mean if (real_amp_mean is not None and fake_amp_mean is not None) else None
    phase_diff = fake_phase_mean - real_phase_mean if (real_phase_mean is not None and fake_phase_mean is not None) else None
    
    # Statistics
    stats = {
        'dataset': dataset_name,
        'num_real_samples': len(real_amps),
        'num_fake_samples': len(fake_amps),
        'real_hf_energy_mean': np.mean(real_hf_energies) if real_hf_energies else 0,
        'real_hf_energy_std': np.std(real_hf_energies) if real_hf_energies else 0,
        'fake_hf_energy_mean': np.mean(fake_hf_energies) if fake_hf_energies else 0,
        'fake_hf_energy_std': np.std(fake_hf_energies) if fake_hf_energies else 0,
        'amp_mean_global_real': np.mean(real_amp_mean) if real_amp_mean is not None else 0,
        'amp_mean_global_fake': np.mean(fake_amp_mean) if fake_amp_mean is not None else 0,
        'phase_mean_global_real': np.mean(real_phase_mean) if real_phase_mean is not None else 0,
        'phase_mean_global_fake': np.mean(fake_phase_mean) if fake_phase_mean is not None else 0,
    }
    
    # Get representative samples
    real_samples = get_representative_samples(dataset_dir, "real", 3, seed=42)
    fake_samples = get_representative_samples(dataset_dir, "fake", 3, seed=42)
    
    # Determine which layers were used
    if use_model:
        layers_used = []
        if activations and ('HFRF_feature' in activations or 'HFRI_feature' in activations):
            if 'HFRF_feature' in activations:
                layers_used.append('HFRF (hfreqC)')
            if 'HFRI_feature' in activations:
                layers_used.append('HFRI (hfreqWH)')
        else:
            # Check for fallback layers
            for key in activations:
                if key.startswith('fallback_'):
                    layers_used.append(key.replace('fallback_', ''))
        layers_used_str = ', '.join(layers_used) if layers_used else 'model activations (fallback)'
    else:
        layers_used_str = 'image-level FFT'
    
    result = {
        'dataset_name': dataset_name,
        'real_amp_mean': real_amp_mean,
        'real_phase_mean': real_phase_mean,
        'fake_amp_mean': fake_amp_mean,
        'fake_phase_mean': fake_phase_mean,
        'amp_diff': amp_diff,
        'phase_diff': phase_diff,
        'stats': stats,
        'real_samples': real_samples,
        'fake_samples': fake_samples,
        'layers_used': layers_used_str
    }
    
    # Clean up hooks (restore original methods)
    if hooks and model is not None:
        for item in hooks:
            if isinstance(item, tuple) and len(item) == 2:
                method_name, original_method = item
                if method_name == 'hook_handles':
                    # Remove hook handles
                    for handle in original_method:
                        handle.remove()
                elif original_method is not None:
                    setattr(model, method_name, original_method)
    
    return result

# ============================================================================
# VISUALIZATION
# ============================================================================
def create_dataset_visualization(result, output_dir, global_amp_range=None, global_phase_range=None):
    """Create visualization for a single dataset."""
    dataset_name = result['dataset_name']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name} - HFR Analysis (N_real={result["stats"]["num_real_samples"]}, N_fake={result["stats"]["num_fake_samples"]})', 
                 fontsize=16, fontweight='bold')
    
    # Determine global ranges if not provided
    if global_amp_range is None:
        amp_min = min(result['real_amp_mean'].min(), result['fake_amp_mean'].min())
        amp_max = max(result['real_amp_mean'].max(), result['fake_amp_mean'].max())
        global_amp_range = (amp_min, amp_max)
    
    if global_phase_range is None:
        phase_min = -np.pi
        phase_max = np.pi
        global_phase_range = (phase_min, phase_max)
    
    # Row 1: Real amplitude, Real phase, Fake amplitude
    im1 = axes[0, 0].imshow(result['real_amp_mean'], cmap='gray', vmin=global_amp_range[0], vmax=global_amp_range[1])
    axes[0, 0].set_title('Real - Amplitude (log magnitude)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='log magnitude')
    
    im2 = axes[0, 1].imshow(result['real_phase_mean'], cmap='seismic', vmin=global_phase_range[0], vmax=global_phase_range[1])
    axes[0, 1].set_title('Real - Phase (radians)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='radians')
    
    im3 = axes[0, 2].imshow(result['fake_amp_mean'], cmap='gray', vmin=global_amp_range[0], vmax=global_amp_range[1])
    axes[0, 2].set_title('Fake - Amplitude (log magnitude)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], label='log magnitude')
    
    # Row 2: Fake phase, Amplitude diff, Phase diff
    im4 = axes[1, 0].imshow(result['fake_phase_mean'], cmap='seismic', vmin=global_phase_range[0], vmax=global_phase_range[1])
    axes[1, 0].set_title('Fake - Phase (radians)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], label='radians')
    
    # Amplitude difference (diverging colormap)
    amp_diff_max = np.abs(result['amp_diff']).max()
    im5 = axes[1, 1].imshow(result['amp_diff'], cmap='RdBu_r', vmin=-amp_diff_max, vmax=amp_diff_max)
    axes[1, 1].set_title('Amplitude Diff (Fake - Real)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], label='difference')
    
    # Phase difference
    phase_diff_max = np.abs(result['phase_diff']).max()
    im6 = axes[1, 2].imshow(result['phase_diff'], cmap='RdBu_r', vmin=-phase_diff_max, vmax=phase_diff_max)
    axes[1, 2].set_title('Phase Diff (Fake - Real)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], label='difference')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{dataset_name}_HFR_vis.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_path}")
    return output_path

def create_examples_grid(result, output_dir):
    """Create grid of example images."""
    dataset_name = result['dataset_name']
    real_samples = result['real_samples']
    fake_samples = result['fake_samples']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'{dataset_name} - Example Images', fontsize=14, fontweight='bold')
    
    # Real samples
    for i, sample in enumerate(real_samples[:3]):
        axes[0, i].imshow(sample)
        axes[0, i].set_title(f'Real {i+1}', fontsize=10)
        axes[0, i].axis('off')
    
    # Fake samples
    for i, sample in enumerate(fake_samples[:3]):
        axes[1, i].imshow(sample)
        axes[1, i].set_title(f'Fake {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    output_path = output_dir / f"{dataset_name}_examples.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_summary_visualization(all_results, output_dir):
    """Create summary comparison across all datasets."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Amplitude Difference (Fake - Real) Across Datasets', fontsize=16, fontweight='bold')
    
    dataset_order = ["FFpp_c40", "FFpp_c23", "CelebDF", "WDF"]
    
    # Compute global range for amplitude differences
    all_amp_diffs = [r['amp_diff'] for r in all_results.values() if r['amp_diff'] is not None]
    if all_amp_diffs:
        global_min = min([ad.min() for ad in all_amp_diffs])
        global_max = max([ad.max() for ad in all_amp_diffs])
        global_range = max(abs(global_min), abs(global_max))
    else:
        global_range = 1.0
    
    for idx, dataset_name in enumerate(dataset_order):
        if dataset_name not in all_results:
            continue
        
        result = all_results[dataset_name]
        if result['amp_diff'] is None:
            continue
        
        im = axes[idx].imshow(result['amp_diff'], cmap='RdBu_r', vmin=-global_range, vmax=global_range)
        axes[idx].set_title(f'{dataset_name}\n(N_real={result["stats"]["num_real_samples"]}, N_fake={result["stats"]["num_fake_samples"]})', 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], label='difference')
    
    plt.tight_layout()
    
    output_path = output_dir / "all_datasets_HFR_summary.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary visualization: {output_path}")
    return output_path

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution function."""
    print("="*60)
    print("FreqNet HFRI/HFRF Block Visualization")
    print("="*60)
    
    # Setup
    output_dir = setup_output_dir(OUTPUT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Process each dataset
    all_results = {}
    all_stats = []
    
    for dataset_name in ["FFpp_c23", "FFpp_c40", "CelebDF", "WDF"]:
        checkpoint_path = CHECKPOINT_PATHS.get(dataset_name)
        dataset_dir = DATASET_DIRS.get(dataset_name)
        
        if not checkpoint_path or not dataset_dir:
            print(f"Skipping {dataset_name}: paths not configured")
            continue
        
        result = process_dataset(dataset_name, dataset_dir, checkpoint_path, device, output_dir)
        if result:
            all_results[dataset_name] = result
            all_stats.append(result['stats'])
    
    if not all_results:
        print("Error: No datasets processed successfully")
        return
    
    # Compute global ranges for visualization
    all_amp_means = [r['real_amp_mean'] for r in all_results.values() if r['real_amp_mean'] is not None]
    all_amp_means.extend([r['fake_amp_mean'] for r in all_results.values() if r['fake_amp_mean'] is not None])
    if all_amp_means:
        global_amp_min = min([am.min() for am in all_amp_means])
        global_amp_max = max([am.max() for am in all_amp_means])
        global_amp_range = (global_amp_min, global_amp_max)
    else:
        global_amp_range = None
    
    global_phase_range = (-np.pi, np.pi)
    
    # Create visualizations
    figure_paths = []
    
    for dataset_name, result in all_results.items():
        # Dataset-specific visualization
        fig_path = create_dataset_visualization(result, output_dir, global_amp_range, global_phase_range)
        figure_paths.append(str(fig_path))
        
        # Examples grid
        ex_path = create_examples_grid(result, output_dir)
        
        # Save raw arrays
        raw_dir = output_dir / "raw_arrays"
        np.save(raw_dir / f"{dataset_name}_real_amp_mean.npy", result['real_amp_mean'])
        np.save(raw_dir / f"{dataset_name}_real_phase_mean.npy", result['real_phase_mean'])
        np.save(raw_dir / f"{dataset_name}_fake_amp_mean.npy", result['fake_amp_mean'])
        np.save(raw_dir / f"{dataset_name}_fake_phase_mean.npy", result['fake_phase_mean'])
        np.save(raw_dir / f"{dataset_name}_amp_diff.npy", result['amp_diff'])
        np.save(raw_dir / f"{dataset_name}_phase_diff.npy", result['phase_diff'])
    
    # Summary visualization
    summary_path = create_summary_visualization(all_results, output_dir)
    figure_paths.append(str(summary_path))
    
    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    stats_csv_path = output_dir / "hfr_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved statistics: {stats_csv_path}")
    
    # Create README
    layers_used = all_results[list(all_results.keys())[0]]['layers_used']
    readme_content = f"""# FreqNet HFR Visualization Results

## Layers Probed
{layers_used}

## Processing Parameters
- num_samples_per_class: {num_samples_per_class}
- image_size: {image_size}
- center_frac: {center_frac}

## Datasets Processed
"""
    for dataset_name, result in all_results.items():
        readme_content += f"""
### {dataset_name}
- Real samples: {result['stats']['num_real_samples']}
- Fake samples: {result['stats']['num_fake_samples']}
- Real HF energy mean: {result['stats']['real_hf_energy_mean']:.4f} ± {result['stats']['real_hf_energy_std']:.4f}
- Fake HF energy mean: {result['stats']['fake_hf_energy_mean']:.4f} ± {result['stats']['fake_hf_energy_std']:.4f}

"""
    
    readme_content += f"""
## Output Files
- Per-dataset visualizations: `{{dataset}}_HFR_vis.png`
- Example images: `{{dataset}}_examples.png`
- Summary comparison: `all_datasets_HFR_summary.png`
- Statistics: `hfr_stats.csv`
- Raw arrays: `raw_arrays/{{dataset}}_{{metric}}.npy`

## Interpretation Notes
- Amplitude (HFRF) shows the log magnitude of FFT features
- Phase (HFRI) shows the phase angle of FFT features
- Differences (Fake - Real) highlight where fake images differ from real images in frequency domain
- Higher HF energy indicates more high-frequency content
"""
    
    readme_path = output_dir / "hfr_readme.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Saved README: {readme_path}")
    
    # Final JSON summary
    summary_json = {
        "figures": figure_paths,
        "stats_csv": str(stats_csv_path),
        "notes": f"HFRI/HFRF hooks used: {layers_used}"
    }
    
    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Figures: {len(figure_paths)}")
    print(f"Statistics: {stats_csv_path}")
    print("\nSummary JSON:")
    print(json.dumps(summary_json, indent=2))

if __name__ == "__main__":
    main()

