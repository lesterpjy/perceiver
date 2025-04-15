#!/usr/bin/env python
"""
Fix script for the data handling issue in LitImageClassifier.
This script modifies the validation_step and step methods to ensure proper data processing.

Usage:
    python fix_data_handling.py
"""

import os
import sys
from pathlib import Path
import re


def fix_data_handling():
    """
    Fix data handling issues in the model code.
    """
    # Find relevant files
    lightning_py_file = find_file("lightning.py", "image_classifier")
    full_train_py_file = find_file("full_train.py")

    if not lightning_py_file:
        print("Error: Could not find lightning.py in the image_classifier directory.")
        return False

    if not full_train_py_file:
        print("Error: Could not find full_train.py.")
        return False

    print(f"Found lightning.py at: {lightning_py_file}")
    print(f"Found full_train.py at: {full_train_py_file}")

    # Create backups
    import shutil

    shutil.copy2(lightning_py_file, f"{lightning_py_file}.bak")
    shutil.copy2(full_train_py_file, f"{full_train_py_file}.bak")
    print(f"Created backups of both files")

    # Fix 1: Modify LitImageClassifier.step to handle both tuple and dict batch formats
    with open(lightning_py_file, "r") as f:
        lightning_content = f.read()

    # Find the step method
    step_pattern = r"def step\(self, batch\):(.*?)return self\.loss_acc\(self\(x\), y\)"
    step_match = re.search(step_pattern, lightning_content, re.DOTALL)

    if not step_match:
        print("Warning: Could not find step method in LitImageClassifier. Cannot update it.")
    else:
        # Create a more robust step method
        new_step = """def step(self, batch):
        # Handle both dict format and tuple format
        if isinstance(batch, dict) and "image" in batch and "label" in batch:
            x, y = batch["image"], batch["label"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Expected dict with 'image'/'label' keys or tuple of (image, label)")
        
        # Ensure x is a tensor
        if not hasattr(x, 'shape'):
            raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")
            
        return self.loss_acc(self(x), y)"""

        # Replace the old step method with the new one
        new_lightning_content = re.sub(step_pattern, new_step, lightning_content, flags=re.DOTALL)

        with open(lightning_py_file, "w") as f:
            f.write(new_lightning_content)

        print("Updated step method in LitImageClassifier to handle different batch formats")

    # Fix 2: Add forward method in LitImageClassifier with input validation
    forward_pattern = r"def forward\(self, x\):(.*?)return self\.model\(x\)"
    if re.search(forward_pattern, lightning_content, re.DOTALL):
        # Replace forward method with a more robust version
        new_forward = """def forward(self, x):
        # Ensure x is a tensor
        if not hasattr(x, 'shape'):
            raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")
            
        return self.model(x)"""

        new_lightning_content = re.sub(forward_pattern, new_forward, new_lightning_content, flags=re.DOTALL)

        with open(lightning_py_file, "w") as f:
            f.write(new_lightning_content)

        print("Updated forward method in LitImageClassifier with input validation")

    # Fix 3: Update validation_step method in EnhancedLitImageClassifier
    with open(full_train_py_file, "r") as f:
        full_train_content = f.read()

    # Find the validation_step method
    val_step_pattern = r"def validation_step\(self, batch, batch_idx\):(.*?)return loss"
    val_step_match = re.search(val_step_pattern, full_train_content, re.DOTALL)

    if not val_step_match:
        print("Warning: Could not find validation_step method in EnhancedLitImageClassifier. Cannot update it.")
    else:
        # Extract and modify the validation_step method
        val_step_content = val_step_match.group(1)

        # Update the line that unpacks the batch
        val_step_content = re.sub(
            r"x, y = batch",
            """# Handle both dict format and tuple format
        if isinstance(batch, dict) and "image" in batch and "label" in batch:
            x, y = batch["image"], batch["label"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Expected dict with 'image'/'label' keys or tuple of (image, label)")
        
        # Ensure x is a tensor
        if not hasattr(x, 'shape'):
            raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")""",
            val_step_content,
        )

        # Reassemble the method
        new_val_step = f"def validation_step(self, batch, batch_idx):{val_step_content}return loss"

        # Replace in the full content
        new_full_train_content = re.sub(val_step_pattern, new_val_step, full_train_content, flags=re.DOTALL)

        # Also update training_step method
        train_step_pattern = r"def training_step\(self, batch, batch_idx\):(.*?)return loss"
        train_step_match = re.search(train_step_pattern, full_train_content, re.DOTALL)

        if train_step_match:
            train_step_content = train_step_match.group(1)

            # Update the line that unpacks the batch
            train_step_content = re.sub(
                r"x, y = batch",
                """# Handle both dict format and tuple format
            if isinstance(batch, dict) and "image" in batch and "label" in batch:
                x, y = batch["image"], batch["label"]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}. Expected dict with 'image'/'label' keys or tuple of (image, label)")
            
            # Ensure x is a tensor
            if not hasattr(x, 'shape'):
                raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")""",
                train_step_content,
            )

            # Reassemble the method
            new_train_step = f"def training_step(self, batch, batch_idx):{train_step_content}return loss"

            # Replace in the full content
            new_full_train_content = re.sub(train_step_pattern, new_train_step, new_full_train_content, flags=re.DOTALL)

            print("Updated training_step method in EnhancedLitImageClassifier with better batch handling")

        # Also update test_step method
        test_step_pattern = (
            r"def test_step\(self, batch, batch_idx\):(.*?)return \{\"loss\": loss, \"preds\": preds, \"targets\": y\}"
        )
        test_step_match = re.search(test_step_pattern, new_full_train_content, re.DOTALL)

        if test_step_match:
            test_step_content = test_step_match.group(1)

            # Update the line that unpacks the batch
            test_step_content = re.sub(
                r"x, y = batch",
                """# Handle both dict format and tuple format
            if isinstance(batch, dict) and "image" in batch and "label" in batch:
                x, y = batch["image"], batch["label"]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}. Expected dict with 'image'/'label' keys or tuple of (image, label)")
            
            # Ensure x is a tensor
            if not hasattr(x, 'shape'):
                raise TypeError(f"Input x must be a tensor with shape attribute, got {type(x)}")""",
                test_step_content,
            )

            # Reassemble the method
            new_test_step = f'def test_step(self, batch, batch_idx):{test_step_content}return {{"loss": loss, "preds": preds, "targets": y}}'

            # Replace in the full content
            new_full_train_content = re.sub(test_step_pattern, new_test_step, new_full_train_content, flags=re.DOTALL)

            print("Updated test_step method in EnhancedLitImageClassifier with better batch handling")

        with open(full_train_py_file, "w") as f:
            f.write(new_full_train_content)

        print("Updated validation_step method in EnhancedLitImageClassifier with better batch handling")

    return True


def find_file(filename, subdir=None):
    """
    Find a file in the project directory structure, optionally looking in a specific subdirectory.
    """
    start_dir = os.getcwd()

    if subdir:
        # Look in directories that match the subdir pattern
        for root, dirs, files in os.walk(start_dir):
            if subdir in root.lower() and filename in files:
                return os.path.join(root, filename)
    else:
        # Look everywhere
        for root, dirs, files in os.walk(start_dir):
            if filename in files:
                return os.path.join(root, filename)

    return None


def main():
    print("=" * 80)
    print("DATA HANDLING FIX FOR PERCEIVER MODEL")
    print("=" * 80)

    if fix_data_handling():
        print("\nFix applied successfully!")
        print("\nThe script has updated the data handling code to:")
        print("1. Ensure proper batch format detection and handling")
        print("2. Add input validation to catch type errors early")
        print("3. Handle both dictionary and tuple batch formats")

        print("\nTo test if the issues are resolved, run your original script:")
        print("    python test_perceiver_local.py")
    else:
        print("\nFailed to apply the fix.")

    print("=" * 80)


if __name__ == "__main__":
    main()
