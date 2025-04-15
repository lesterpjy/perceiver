#!/usr/bin/env python
"""
Local testing script for Perceiver IO that runs on CPU.
This script allows you to verify your training pipeline works before submitting to the cluster.

Usage:
    python test_perceiver_local.py

This will use the local_config.yaml file created alongside this script.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Try importing required packages, install if missing
try:
    import torch
    import pytorch_lightning as pl
except ImportError:
    print("Installing required packages...")
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "pytorch-lightning>=2.0",
            "torchmetrics>=0.9",
            "einops>=0.4",
            "loguru>=0.6.0",
            "pyyaml>=6.0",
        ]
    )
    import torch
    import pytorch_lightning as pl

# Add the project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import required modules - this assumes we're running from project root
try:
    from examples.training.img_clf.full_train import (
        setup_logger,
        load_config,
        update_config_with_args,
        setup_output_dir,
        log_config,
        setup_callbacks,
        create_model_config,
        EnhancedLitImageClassifier,
        create_trainer,
        generate_report,
    )
    from perceiver.data.vision import MNISTDataModule
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

from loguru import logger


def verify_setup():
    """Verifies that the environment is set up correctly for local testing"""
    print("Verifying local setup...")

    # Check PyTorch installation and device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Check for necessary directories
    for dir_path in ["perceiver", "examples"]:
        if not os.path.isdir(dir_path):
            print(f"WARNING: Expected directory '{dir_path}' not found!")
            print("Make sure you're running this script from the project root directory.")

    # Check for setup.py (needed for editable installs)
    if not os.path.exists("setup.py"):
        print("Creating setup.py for editable install...")
        setup_py_content = """#!/usr/bin/env python
                        from setuptools import setup, find_packages

                        if __name__ == "__main__":
                            setup(
                                name="perceiver-io",
                                version="0.11.1",
                                description="Perceiver IO",
                                author="Martin Krasser, Christoph Stumpf",
                                packages=find_packages(),
                                python_requires=">=3.8,<3.11",
                            )
                        """
        with open("setup.py", "w") as f:
            f.write(setup_py_content)

    # Enable tensor cores with medium precision
    torch.set_float32_matmul_precision("medium")
    print("Tensor cores enabled with medium precision")


def main():
    """Main function for local testing"""
    print("=" * 80)
    print("PERCEIVER IO LOCAL CPU TESTING")
    print("=" * 80)

    # Verify setup
    verify_setup()

    # local config
    config_path = "./local_config.yaml"

    # Load configuration
    config = load_config(config_path)

    # Set up output directory
    output_dir = setup_output_dir(config)

    # Set up logger
    setup_logger(config, output_dir)
    logger.info(f"Starting Perceiver IO local testing with config from {config_path}")

    # Log configuration
    log_config(config, logger)

    # Set seed for reproducibility
    seed = config["experiment"]["seed"]
    if seed is not None:
        pl.seed_everything(seed)
        logger.info(f"Setting random seed to {seed}")

    # Initialize data module
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    data = MNISTDataModule(batch_size=batch_size, num_workers=num_workers, pin_memory=config["data"]["pin_memory"])
    logger.info(f"Initialized MNIST data module with batch_size={batch_size}, num_workers={num_workers}")

    # Set up callbacks
    callbacks, checkpoint_callback, compute_callback = setup_callbacks(config, output_dir)

    # Create model configuration
    model_config = create_model_config(config, data)

    # Initialize enhanced model
    lit_model = EnhancedLitImageClassifier.create(model_config, config)
    logger.info("Model initialized successfully")

    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lit_model.parameters())
    logger.info(f"Model has {trainable_params:,}/{total_params:,} trainable parameters")

    # Create trainer
    trainer = create_trainer(config, output_dir, callbacks)
    logger.info(f"Trainer created with {config['training']['devices']} devices")

    try:
        # Train model
        logger.info("=" * 50)
        logger.info("STARTING TRAINING")
        logger.info("=" * 50)
        trainer.fit(lit_model, datamodule=data)

        # Test model
        logger.info("=" * 50)
        logger.info("STARTING TESTING")
        logger.info("=" * 50)
        try:
            # Try to run test with the datamodule
            test_result = trainer.test(lit_model, datamodule=data)
        except Exception as e:
            if "test_dataloader" in str(e):
                logger.warning("Test dataloader not available. Using validation dataloader for testing.")
                # Use the validation dataloader for testing
                test_result = trainer.test(lit_model, val_dataloaders=trainer.val_dataloaders)
            else:
                # Re-raise other exceptions
                raise

        # Generate and save performance report
        compute_callback.save_stats(output_dir)
        report_path = generate_report(trainer, lit_model, compute_callback, output_dir)

        logger.info(f"Local testing complete! Full report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        logger.error(traceback.format_exc())

    print("\n" + "=" * 80)
    print("LOCAL TESTING SUMMARY")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    print("\nIf the script ran without errors, your Perceiver IO setup is working correctly!")
    print("You can now submit your job to the cluster.")
    print("=" * 80)


if __name__ == "__main__":
    main()
