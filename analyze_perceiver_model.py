#!/usr/bin/env python
"""
Debug and analysis tool for Perceiver IO model.
This script inspects the model structure, parameters, and initial behavior.

Usage:
    python analyze_perceiver_model.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import yaml
    from perceiver.data.vision import MNISTDataModule
    from perceiver.model.vision.image_classifier import (
        ImageEncoderConfig,
        LitImageClassifier,
        PerceiverClassifierConfig,
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(config, data):
    """Create model configuration from config dictionary"""
    encoder_config = ImageEncoderConfig(
        image_shape=data.image_shape,
        num_frequency_bands=config["model"]["encoder"]["num_frequency_bands"],
        num_cross_attention_layers=config["model"]["encoder"]["num_cross_attention_layers"],
        num_cross_attention_heads=config["model"]["encoder"]["num_cross_attention_heads"],
        num_self_attention_blocks=config["model"]["encoder"]["num_self_attention_blocks"],
        num_self_attention_layers_per_block=config["model"]["encoder"]["num_self_attention_layers_per_block"],
        first_cross_attention_layer_shared=config["model"]["encoder"]["first_cross_attention_layer_shared"],
        first_self_attention_block_shared=config["model"]["encoder"]["first_self_attention_block_shared"],
        dropout=config["model"]["encoder"]["dropout"],
        init_scale=config["model"]["encoder"]["init_scale"],
    )

    model_config = PerceiverClassifierConfig(
        encoder=encoder_config,
        num_latents=config["model"]["perceiver"]["num_latents"],
        num_latent_channels=config["model"]["perceiver"]["num_latent_channels"],
        num_classes=config["model"]["perceiver"]["num_classes"],
    )

    return model_config


def analyze_model_structure(model):
    """Analyze and print the model structure"""
    print("\n" + "=" * 80)
    print("MODEL STRUCTURE ANALYSIS")
    print("=" * 80)

    # Print high-level structure
    print("\nHigh-level components:")
    print("-" * 80)
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"{name}: {child.__class__.__name__} - Params: {params:,} (Trainable: {trainable:,})")

    # Print detailed structure
    print("\nDetailed model structure:")
    print("-" * 80)

    def print_module_structure(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            params = sum(p.numel() for p in child.parameters())
            trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)

            if params > 0:
                print(f"{full_name}: {child.__class__.__name__} - Params: {params:,} (Trainable: {trainable:,})")

            # Recursively print children
            print_module_structure(child, full_name)

    print_module_structure(model)

    return


def analyze_parameter_gradients(model):
    """Analyze which parameters have gradients enabled"""
    print("\n" + "=" * 80)
    print("PARAMETER GRADIENT ANALYSIS")
    print("=" * 80)

    # Count trainable vs non-trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percent trainable: {trainable_params/total_params*100:.2f}%")

    # Check for parameters without gradients
    if trainable_params < total_params:
        print("\nParameters with gradients disabled:")
        print("-" * 80)
        print(f"{'Parameter':<60} {'Shape':<20} {'Size':<10}")
        print("-" * 80)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"{name:<60} {str(list(param.shape)):<20} {param.numel():<10,}")

    return trainable_params, total_params


def test_forward_pass(model, data):
    """Test a forward pass with the model"""
    print("\n" + "=" * 80)
    print("FORWARD PASS TEST")
    print("=" * 80)

    # Get a batch of data
    train_dataloader = data.train_dataloader()
    batch = next(iter(train_dataloader))
    x, y = batch

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    # Run forward pass
    try:
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if start_time:
                start_time.record()

            output = model(x)

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                print(f"Forward pass time: {elapsed_time:.2f} ms")

        print(f"Output shape: {output.shape}")
        print(f"Output contains NaN: {torch.isnan(output).any().item()}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print("Forward pass successful!")

        # Check if outputs make sense for the task
        predictions = torch.argmax(output, dim=1)
        print(f"Predicted classes distribution: {torch.bincount(predictions)}")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_backward_pass(model, data):
    """Test a backward pass (optimization step) with the model"""
    print("\n" + "=" * 80)
    print("BACKWARD PASS TEST")
    print("=" * 80)

    # Get a batch of data
    train_dataloader = data.train_dataloader()
    batch = next(iter(train_dataloader))
    x, y = batch

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run forward and backward pass
    try:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(output, y)
        print(f"Initial loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
        print(f"Gradient norm: {grad_norm.item():.4f}")

        # Check for NaN gradients
        has_nan_grads = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        print(f"NaN gradients detected: {has_nan_grads}")

        # Optimizer step
        optimizer.step()

        # Compute loss after update
        with torch.no_grad():
            output_after = model(x)
            loss_after = torch.nn.functional.cross_entropy(output_after, y)
            print(f"Loss after update: {loss_after.item():.4f}")
            print(f"Loss change: {loss_after.item() - loss.item():.4f}")

        print("Backward pass successful!")
        return True

    except Exception as e:
        print(f"Error during backward pass: {e}")
        import traceback

        traceback.print_exc()
        return False


def fix_model_if_needed(model):
    """Apply fixes to the model if needed"""
    print("\n" + "=" * 80)
    print("MODEL FIXES")
    print("=" * 80)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable_params == 0:
        print("CRITICAL ISSUE: No trainable parameters found!")
        print("Applying fix: Enabling gradients for all parameters...")

        for param in model.parameters():
            param.requires_grad = True

        trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"After fix: {trainable_params_after:,} trainable parameters")

        if trainable_params_after > 0:
            print("Fix successful!")
        else:
            print("Fix failed! Still no trainable parameters.")
    else:
        print("No critical issues detected that require fixing.")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Perceiver IO model")
    parser.add_argument(
        "--config",
        type=str,
        default="local_config.yaml",
        help="Path to configuration file (default: local_config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("PERCEIVER IO MODEL ANALYZER")
    print("=" * 80)

    print(f"Using configuration file: {args.config}")

    # Load configuration
    config = load_config(args.config)

    # Set up data module
    data = MNISTDataModule(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    data.prepare_data()
    data.setup()

    # Create model configuration
    model_config = create_model_config(config, data)

    # Create model
    model = LitImageClassifier(model_config)

    # Analyze model structure
    analyze_model_structure(model)

    # Analyze parameter gradients
    trainable_params, total_params = analyze_parameter_gradients(model)

    # Fix model if needed
    if trainable_params == 0:
        model = fix_model_if_needed(model)

    # Test forward pass
    forward_pass_success = test_forward_pass(model, data)

    # Test backward pass if forward pass was successful
    if forward_pass_success and trainable_params > 0:
        backward_pass_success = test_backward_pass(model, data)

    # Print final summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"Model architecture: Perceiver IO")
    print(f"Dataset: MNIST")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    if trainable_params == 0:
        print("\nCRITICAL ISSUE: No trainable parameters in the model!")
        print("Fix this by adding 'param.requires_grad = True' for all parameters.")
    elif trainable_params / total_params < 0.5:
        print("\nWARNING: Less than 50% of parameters are trainable.")
        print("This might be intentional, but could indicate an issue.")

    if forward_pass_success:
        print("\nForward pass: SUCCESS")
    else:
        print("\nForward pass: FAILED - Check errors above")

    if "backward_pass_success" in locals():
        if backward_pass_success:
            print("Backward pass: SUCCESS")
        else:
            print("Backward pass: FAILED - Check errors above")

    print("\nRecommendation:")
    if (
        trainable_params == 0
        or not forward_pass_success
        or ("backward_pass_success" in locals() and not backward_pass_success)
    ):
        print("- Fix the issues above before running training")
    else:
        print("- Model looks good! You can proceed with training")

    print("=" * 80)


if __name__ == "__main__":
    main()
