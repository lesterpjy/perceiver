"""
Configuration utilities for Perceiver IO training.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path where to save the configuration
    """
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config_with_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with command line arguments.

    Args:
        config: Base configuration dictionary
        args: Arguments from command line

    Returns:
        Updated configuration dictionary
    """
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()

    # Update nested values based on dot notation in keys
    for key, value in args.items():
        if value is None:
            continue

        if "." in key:
            # Handle nested keys like 'model.encoder.num_latents'
            parts = key.split(".")
            current = updated_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Handle top-level keys
            updated_config[key] = value

    return updated_config


def setup_output_dir(config: Dict[str, Any]) -> str:
    """
    Set up the output directory based on the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the output directory
    """
    import datetime

    # Get the base output directory from the config
    base_dir = config["experiment"]["output_dir"]

    # Create a timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config["experiment"]["name"]
    output_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")

    # Create the directory
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Save the configuration to the output directory
    save_config(config, os.path.join(output_dir, "config.yaml"))

    return output_dir


def log_config(config: Dict[str, Any], logger):
    """
    Log the configuration using the provided logger.

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Configuration:")
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")

    # Log model architecture
    logger.info("Model architecture:")
    logger.info(f"  Num latents: {config['model']['perceiver']['num_latents']}")
    logger.info(f"  Num latent channels: {config['model']['perceiver']['num_latent_channels']}")
    logger.info(f"  Num classes: {config['model']['perceiver']['num_classes']}")
    logger.info(f"  Num frequency bands: {config['model']['encoder']['num_frequency_bands']}")
    logger.info(f"  Cross attention layers: {config['model']['encoder']['num_cross_attention_layers']}")
    logger.info(f"  Self attention blocks: {config['model']['encoder']['num_self_attention_blocks']}")

    # Log training settings
    logger.info("Training settings:")
    logger.info(f"  Max epochs: {config['training']['max_epochs']}")
    logger.info(f"  Batch size: {config['data']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['optimizer']['lr']}")
    logger.info(f"  Weight decay: {config['training']['optimizer']['weight_decay']}")
    logger.info(f"  Devices: {config['training']['devices']}")
    logger.info(f"  Precision: {config['training']['precision']}")


def create_optimizer(config: Dict[str, Any], parameters):
    """
    Create optimizer based on configuration.

    Args:
        config: Configuration dictionary with optimizer settings
        parameters: Model parameters to optimize

    Returns:
        Optimizer instance
    """
    from torch.optim import AdamW, SGD, Adam

    optimizer_name = config["training"]["optimizer"]["name"].lower()
    lr = config["training"]["optimizer"]["lr"]
    weight_decay = config["training"]["optimizer"]["weight_decay"]

    if optimizer_name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_lr_scheduler(config: Dict[str, Any], optimizer):
    """
    Create learning rate scheduler based on configuration.

    Args:
        config: Configuration dictionary with lr_scheduler settings
        optimizer: Optimizer instance

    Returns:
        Learning rate scheduler instance and configuration dictionary
    """
    from perceiver.scripts.lrs import ConstantWithWarmupLR

    scheduler_name = config["training"]["lr_scheduler"]["name"].lower()

    if scheduler_name == "constant_with_warmup":
        warmup_steps = config["training"]["lr_scheduler"]["warmup_steps"]
        scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=warmup_steps)
        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler_config
