"""
Enhanced Train-Eval-Test Script for Perceiver IO.
Uses config files and loguru for logging.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import examples.training  # noqa: F401
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
from fvcore.nn import FlopCountAnalysis, flop_count_table

from perceiver.data.vision import MNISTDataModule
from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.vision.image_classifier import (
    ImageClassifierConfig,
    ImageEncoderConfig,
    LitImageClassifier,
    PerceiverClassifierConfig,
)

# Import custom config utilities
from .config_utils import (
    load_config,
    save_config,
    update_config_with_args,
    setup_output_dir,
    log_config,
    create_optimizer,
    create_lr_scheduler,
)


# Configure loguru logger
def setup_logger(config, output_dir=None):
    """Configure loguru logger based on config."""
    # Remove default handler
    logger.remove()

    # Add console handler with appropriate level
    log_level = config["logging"]["level"]
    logger.add(sys.stderr, level=log_level)

    # Add file handler if enabled
    if config["logging"]["file"] and output_dir:
        log_path = os.path.join(output_dir, "logs", "train.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.add(log_path, rotation="10 MB", level=log_level)

    logger.info(f"Logger configured with level: {log_level}")
    return logger


# Define a parameter check callback to fix the no trainable params issue
class ParamCheckCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.has_checked = False

    def on_fit_start(self, trainer, pl_module):
        if not self.has_checked:
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in pl_module.parameters())

            logger.info(f"Model has {total_params:,} total parameters")
            logger.info(f"Model has {trainable_params:,} trainable parameters")

            if trainable_params == 0:
                logger.warning("MODEL HAS NO TRAINABLE PARAMETERS! Fixing...")
                # Enable gradients for all parameters
                for name, param in pl_module.named_parameters():
                    param.requires_grad = True
                    logger.debug(f"Enabled gradients for {name}")

                # Check again
                trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
                logger.info(f"After fix: Model has {trainable_params:,} trainable parameters")

                if trainable_params == 0:
                    logger.error("CRITICAL: Still no trainable parameters after fix!")
                    # Try to debug further
                    for name, module in pl_module.named_modules():
                        logger.debug(
                            f"Module: {name}, Trainable params: {sum(p.numel() for p in module.parameters() if p.requires_grad)}"
                        )

            self.has_checked = True


# Custom callback to track compute time and resources
class ComputeStatsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None
        self.epoch_start_times = []
        self.epoch_end_times = []
        self.stats = {}

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()
        logger.info("Starting training process, tracking compute stats...")

    def on_fit_end(self, trainer, pl_module):
        self.end_time = time.time()
        self.stats["total_time_seconds"] = self.end_time - self.start_time
        self.stats["total_time_formatted"] = str(timedelta(seconds=self.stats["total_time_seconds"]))

        # Calculate epoch times
        epoch_times = []
        for start, end in zip(self.epoch_start_times, self.epoch_end_times):
            epoch_times.append(end - start)

        self.stats["epoch_times"] = epoch_times
        self.stats["avg_epoch_time"] = np.mean(epoch_times) if epoch_times else 0
        self.stats["avg_epoch_time_formatted"] = str(timedelta(seconds=self.stats["avg_epoch_time"]))

        logger.info(f"Training completed in {self.stats['total_time_formatted']}")
        logger.info(f"Average epoch time: {self.stats['avg_epoch_time_formatted']}")

        # Calculate FLOPS for a single forward pass
        if hasattr(pl_module, "example_input_array"):
            example_input = pl_module.example_input_array
        else:
            # Create dummy input with the right shape
            batch_size = 1
            image_shape = pl_module.config.encoder.image_shape
            example_input = torch.randn(batch_size, *image_shape).to(pl_module.device)

        logger.info("Calculating FLOPS for model...")
        with torch.no_grad():
            flops = FlopCountAnalysis(pl_module, example_input)
            self.stats["flops_per_forward_pass"] = flops.total()
            self.stats["flops_table"] = flop_count_table(flops)
            logger.info(f"FLOPS per forward pass: {self.stats['flops_per_forward_pass']:,}")

        # Estimate total FLOPS
        num_batches = len(trainer.train_dataloader) if hasattr(trainer, "train_dataloader") else 0
        num_epochs = trainer.current_epoch + 1
        self.stats["estimated_total_flops"] = (
            self.stats["flops_per_forward_pass"] * num_batches * num_epochs * 2
        )  # *2 for forward and backward
        logger.info(f"Estimated total FLOPS: {self.stats['estimated_total_flops']:,}")

        # Memory stats
        if torch.cuda.is_available():
            self.stats["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.stats["peak_gpu_memory_gb"] = self.stats["peak_gpu_memory_mb"] / 1024
            logger.info(f"Peak GPU memory: {self.stats['peak_gpu_memory_mb']:.2f} MB")

    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start_times.append(time.time())
        logger.info(f"Starting epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")

    def on_epoch_end(self, trainer, pl_module):
        self.epoch_end_times.append(time.time())
        epoch_time = self.epoch_end_times[-1] - self.epoch_start_times[-1]
        logger.info(f"Epoch {trainer.current_epoch + 1} completed in {timedelta(seconds=epoch_time)}")

    @rank_zero_only
    def save_stats(self, output_dir):
        """Save the compute statistics to a JSON file"""
        stats_path = os.path.join(output_dir, "compute_stats.json")
        # Convert any non-serializable objects to strings
        serializable_stats = {}
        for k, v in self.stats.items():
            if isinstance(v, (int, float, str, list, dict, bool)) or v is None:
                serializable_stats[k] = v
            else:
                serializable_stats[k] = str(v)

        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Compute statistics saved to {stats_path}")
        return stats_path


class EnhancedLitImageClassifier(LitImageClassifier):
    def __init__(
        self,
        encoder,
        num_latents=None,
        num_latent_channels=None,
        num_classes=None,
        activation_checkpointing=False,
        activation_offloading=False,
        params=None,
        **kwargs,
    ):
        # Call parent constructor
        super().__init__(
            encoder=encoder,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            num_classes=num_classes,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
            params=params,
            **kwargs,
        )

        # Use latest torchmetrics API
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes)

        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

        # CHANGE: Use self.hparams.encoder instead of self.config.encoder
        self.example_input_array = torch.zeros(1, *self.hparams.encoder.image_shape)

        logger.debug(f"Enhanced model initialized with num_classes={self.hparams.num_classes}")

        # Log trainable parameter count
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {trainable_params:,}/{total_params:,} trainable parameters")

    def training_step(self, batch, batch_idx):
        # Handle both dict format and tuple format
        if isinstance(batch, dict) and "image" in batch and "label" in batch:
            x, y = batch["image"], batch["label"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(
                f"Unexpected batch format: {type(batch)}. Expected dict with 'image'/'label' keys or tuple of (image, label)"
            )

        # Ensure x is a tensor
        if not hasattr(x, "shape"):
            raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update and log metrics
        self.train_acc(preds, y)
        self.train_f1(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        if batch_idx == 0 and self.current_epoch == 0:
            logger.debug(f"First batch shape: {x.shape}, predictions shape: {logits.shape}")

        return loss

    def validation_step(self, batch, batch_idx):
        # Handle both dict format and tuple format
        if isinstance(batch, dict) and "image" in batch and "label" in batch:
            x, y = batch["image"], batch["label"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(
                f"Unexpected batch format: {type(batch)}. Expected dict with 'image'/'label' keys or tuple of (image, label)"
            )

        # Ensure x is a tensor
        if not hasattr(x, "shape"):
            raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update and log metrics
        self.val_acc(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Handle both dict format and tuple format
        if isinstance(batch, dict) and "image" in batch and "label" in batch:
            x, y = batch["image"], batch["label"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(
                f"Unexpected batch format: {type(batch)}. Expected dict with 'image'/'label' keys or tuple of (image, label)"
            )

        # Ensure x is a tensor
        if not hasattr(x, "shape"):
            raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update and log metrics
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_confusion(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": y}

    def configure_optimizers(self):
        """
        Configure optimizers based on the configuration.
        Overrides the default implementation.
        """
        # FIX: Ensure we're passing only parameters that require gradients to the optimizer
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning(
                "No trainable parameters found when configuring optimizer! Enabling gradients for all parameters."
            )
            for param in self.parameters():
                param.requires_grad = True
            trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = create_optimizer(self.hparams, trainable_params)
        scheduler_config = create_lr_scheduler(self.hparams, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    @classmethod
    def create(cls, config, hparams=None):
        """
        Create the model with the given configuration and hyperparameters.
        Properly passes arguments to match the parent class constructor.
        """
        # Use the parent's create method to ensure correct initialization
        model = super().create(config)

        # Then add your customizations
        if hparams:
            model.hparams.update(hparams)

        # Ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True

        return model


# Function to generate a performance report
def generate_report(trainer, model, compute_callback, output_dir):
    """Generate a comprehensive performance report"""
    report = {}

    # Training stats
    report["epochs_completed"] = trainer.current_epoch + 1
    report["max_epochs"] = trainer.max_epochs
    report["early_stopping"] = trainer.current_epoch + 1 < trainer.max_epochs

    # Performance metrics from the trainer
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        if isinstance(v, torch.Tensor):
            metrics[k] = float(v.detach().cpu().numpy())
        else:
            metrics[k] = v
    report["metrics"] = metrics

    # Compute stats
    report["compute_stats"] = compute_callback.stats

    # Save report to JSON
    report_path = os.path.join(output_dir, "performance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Generate confusion matrix plot if available
    if hasattr(model, "test_confusion") and hasattr(model.test_confusion, "confmat"):
        cm = model.test_confusion.confmat.detach().cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

    # Print summary
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE - PERFORMANCE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Epochs completed: {report['epochs_completed']}/{report['max_epochs']}")

    metrics = report["metrics"]
    if "test_acc" in metrics:
        logger.info(f"Final test accuracy: {metrics['test_acc']:.4f}")
    if "test_f1" in metrics:
        logger.info(f"Final test F1 score: {metrics['test_f1']:.4f}")

    compute_stats = report["compute_stats"]
    logger.info("\nCOMPUTE RESOURCES:")
    logger.info(f"Total training time: {compute_stats['total_time_formatted']}")
    logger.info(f"Average epoch time: {compute_stats.get('avg_epoch_time_formatted', 'N/A')}")

    if "peak_gpu_memory_mb" in compute_stats:
        logger.info(f"Peak GPU memory: {compute_stats['peak_gpu_memory_mb']:.2f} MB")

    if "estimated_total_flops" in compute_stats:
        flops = compute_stats["estimated_total_flops"]
        if flops > 1e12:
            logger.info(f"Estimated total FLOPS: {flops / 1e12:.4f} TFLOPS")
        else:
            logger.info(f"Estimated total FLOPS: {flops / 1e9:.4f} GFLOPS")

    logger.info(f"\nFull reports saved to: {output_dir}")
    logger.info("=" * 50)

    return report_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a Perceiver IO model with configuration file")
    parser.add_argument("--config", type=str, default="perceiver_config.yaml", help="Path to the configuration file")
    parser.add_argument("--experiment.name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--model.perceiver.num_latents", type=int, default=None, help="Override number of latents")
    parser.add_argument(
        "--model.perceiver.num_latent_channels", type=int, default=None, help="Override number of latent channels"
    )
    parser.add_argument("--training.max_epochs", type=int, default=None, help="Override maximum number of epochs")
    parser.add_argument("--training.optimizer.lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--training.devices", type=int, default=None, help="Override number of devices")
    parser.add_argument("--trainer.strategy", type=str, default=None, help="Override training strategy")
    args = parser.parse_args()

    # Convert args to dict for config update
    args_dict = {}
    for arg, value in vars(args).items():
        if arg != "config" and value is not None:
            args_dict[arg] = value

    return args.config, args_dict


def create_trainer(config, output_dir, callbacks):
    """Create a PyTorch Lightning trainer based on configuration"""
    # Set up loggers
    loggers = []
    if config["logging"]["tensorboard"]:
        tb_logger = TensorBoardLogger(save_dir=os.path.join(output_dir, "logs"), name="tensorboard")
        loggers.append(tb_logger)

    if config["logging"]["csv"]:
        csv_logger = CSVLogger(save_dir=os.path.join(output_dir, "logs"), name="csv_logs")
        loggers.append(csv_logger)

    # FIX: Set float32 matmul precision for tensor cores
    logger.info("Enabling tensor cores with medium float32 precision")
    torch.set_float32_matmul_precision("medium")

    # Set up trainer configuration
    trainer_config = {
        "accelerator": config["training"]["accelerator"],
        "devices": config["training"]["devices"],
        "max_epochs": config["training"]["max_epochs"],
        "callbacks": callbacks,
        "logger": loggers,
        "deterministic": config["training"]["deterministic"],
    }

    # Set up strategy - FIX: Use auto for single GPU and DDP with find_unused_parameters for multi-GPU
    if config["training"]["devices"] > 1:
        if config["training"]["strategy"] == "ddp_static":
            logger.info("Using DDP static strategy with find_unused_parameters=True")
            trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True, static_graph=True)
        else:
            logger.info("Using DDP strategy with find_unused_parameters=True")
            trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)
    else:
        logger.info("Using 'auto' strategy for single GPU training")
        trainer_config["strategy"] = "auto"

    # Set precision if specified
    if "precision" in config["training"]:
        trainer_config["precision"] = config["training"]["precision"]

    return pl.Trainer(**trainer_config)


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


def setup_callbacks(config, output_dir):
    """Set up callbacks based on configuration"""
    callbacks = []

    # FIX: Add parameter check callback first
    param_check_callback = ParamCheckCallback()
    callbacks.append(param_check_callback)

    # Add checkpoint callback
    if config["training"]["checkpoint"]["save_top_k"] > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, config["training"]["checkpoint"]["dirpath"]),
            filename=config["training"]["checkpoint"]["filename"],
            save_top_k=config["training"]["checkpoint"]["save_top_k"],
            monitor=config["training"]["checkpoint"]["monitor"],
            mode=config["training"]["checkpoint"]["mode"],
        )
        callbacks.append(checkpoint_callback)

    # Add early stopping callback
    if config["training"]["early_stopping"]["enabled"]:
        early_stop_callback = EarlyStopping(
            monitor=config["training"]["early_stopping"]["monitor"],
            patience=config["training"]["early_stopping"]["patience"],
            mode=config["training"]["early_stopping"]["mode"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
        )
        callbacks.append(early_stop_callback)

    # Add compute stats callback
    compute_callback = ComputeStatsCallback()
    callbacks.append(compute_callback)

    # Add LR monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Add progress bar
    progress_bar = TQDMProgressBar(refresh_rate=20)
    callbacks.append(progress_bar)

    return callbacks, checkpoint_callback, compute_callback


# Main function to coordinate the train-eval-test process
def main():
    # Parse command line arguments
    config_path, args_dict = parse_args()

    # Load configuration
    config = load_config(config_path)

    # Update configuration with command line arguments
    if args_dict:
        config = update_config_with_args(config, args_dict)

    # Set up output directory
    output_dir = setup_output_dir(config)

    # Set up logger
    setup_logger(config, output_dir)
    logger.info(f"Starting Perceiver IO training with config from {config_path}")

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
    logger.info("Creating model configuration...")
    logger.info(f"Model configuration: {model_config}")

    # Initialize enhanced model
    lit_model = EnhancedLitImageClassifier.create(model_config, config)
    logger.info("Model initialized successfully")

    # Create trainer
    trainer = create_trainer(config, output_dir, callbacks)
    logger.info(f"Trainer created with {config['training']['devices']} devices")

    # Try training with DDP first
    try:
        # Train model
        logger.info("=" * 50)
        logger.info("STARTING TRAINING")
        logger.info("=" * 50)
        trainer.fit(lit_model, datamodule=data)
    except RuntimeError as e:
        error_msg = str(e)
        if "DistributedDataParallel is not needed when a module doesn't have any parameter" in error_msg:
            logger.error("DDP error detected: No trainable parameters. Switching to single GPU training...")

            # Enable gradients for all parameters
            for param in lit_model.parameters():
                param.requires_grad = True

            # Create new trainer with single GPU
            config["training"]["devices"] = 1
            config["training"]["strategy"] = "auto"
            trainer = create_trainer(config, output_dir, callbacks)
            logger.info("Created new trainer with single GPU and auto strategy")

            # Try training again
            trainer.fit(lit_model, datamodule=data)
        else:
            # Re-raise other errors
            logger.error(f"Unhandled error: {error_msg}")
            raise

    # Test model using best checkpoint
    logger.info("=" * 50)
    logger.info("STARTING TESTING")
    logger.info("=" * 50)

    if config["testing"]["use_best_checkpoint"] and hasattr(checkpoint_callback, "best_model_path"):
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Testing using best checkpoint: {best_model_path}")
            test_result = trainer.test(ckpt_path=best_model_path, datamodule=data)
        else:
            logger.warning("No checkpoint found, testing with current model")
            test_result = trainer.test(datamodule=data)
    else:
        logger.info("Testing with current model")
        test_result = trainer.test(datamodule=data)

    # Generate and save performance report
    compute_callback.save_stats(output_dir)
    report_path = generate_report(trainer, lit_model, compute_callback, output_dir)

    # Export model logs to CSV for easier analysis
    if trainer.logger:
        for logger_inst in trainer.logger:
            if (
                isinstance(logger_inst, CSVLogger)
                and hasattr(logger_inst, "experiment")
                and hasattr(logger_inst.experiment, "metrics")
            ):
                metrics_df = pd.DataFrame(logger_inst.experiment.metrics)
                metrics_df.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)
                logger.info(f"All metrics saved to {output_dir}/all_metrics.csv")

    logger.info(f"Training complete! Full report saved to: {report_path}")
    return output_dir, report_path


if __name__ == "__main__":
    main()
