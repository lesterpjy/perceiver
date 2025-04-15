#!/usr/bin/env python
"""
Fix script for the MNISTDataModule to add the missing test_dataloader method.

Usage:
    python fix_mnist_datamodule.py
"""

import os
import sys
from pathlib import Path
import re


def fix_mnist_datamodule():
    """
    Add the missing test_dataloader method to MNISTDataModule.
    """
    # Find the mnist.py file
    mnist_py_file = find_file("mnist.py")
    if not mnist_py_file:
        print("Error: Could not find mnist.py file.")
        return False

    print(f"Found mnist.py at: {mnist_py_file}")

    # Create backup
    backup_file = str(mnist_py_file) + ".bak"
    import shutil

    shutil.copy2(mnist_py_file, backup_file)
    print(f"Created backup at: {backup_file}")

    # Read the file
    with open(mnist_py_file, "r") as f:
        content = f.read()

    # Check if the class already has a test_dataloader method
    if "def test_dataloader(self" in content:
        print("test_dataloader method already exists. No fix needed.")
        return True

    # Find the MNISTDataModule class
    class_pattern = r"class MNISTDataModule\((.*?)\):"
    class_match = re.search(class_pattern, content)

    if not class_match:
        print("Error: Could not find MNISTDataModule class.")
        return False

    # Find the end of the val_dataloader method
    val_dataloader_pattern = r"def val_dataloader\(self\)(.*?)return val_loader"
    val_dataloader_match = re.search(val_dataloader_pattern, content, re.DOTALL)

    if not val_dataloader_match:
        print("Error: Could not find val_dataloader method.")
        return False

    # Get the indentation level
    val_dataloader_content = val_dataloader_match.group(1)
    indentation = ""
    for line in val_dataloader_content.split("\n"):
        if line.strip() and not line.strip().startswith("#"):
            indentation = re.match(r"(\s*)", line).group(1)
            break

    # Create the test_dataloader method based on the val_dataloader method
    test_dataloader_method = f"""
    def test_dataloader(self):
        \"\"\"
        Implement the test dataloader method required by PyTorch Lightning.
        For simplicity, we're reusing the validation set as the test set.
        \"\"\"
{val_dataloader_content}
        test_loader = val_loader  # Reuse validation set as test set
        return test_loader
"""

    # Find a good insertion point - after the val_dataloader method
    val_dataloader_end = content.find("return val_loader") + len("return val_loader")
    before_content = content[:val_dataloader_end]
    after_content = content[val_dataloader_end:]

    # Insert the new method
    new_content = before_content + test_dataloader_method + after_content

    # Write the modified content
    with open(mnist_py_file, "w") as f:
        f.write(new_content)

    print(f"Successfully added test_dataloader method to MNISTDataModule in {mnist_py_file}")

    # Alternate approach: create a custom subclass if we can't modify the original
    if not os.path.exists("./custom_mnist.py"):
        custom_datamodule = """#!/usr/bin/env python
\"\"\"
Custom extension of MNISTDataModule that adds the required test_dataloader method.
\"\"\"

from perceiver.data.vision import MNISTDataModule as OriginalMNISTDataModule

class ExtendedMNISTDataModule(OriginalMNISTDataModule):
    \"\"\"
    Extended version of MNISTDataModule that adds the test_dataloader method.
    \"\"\"
    
    def test_dataloader(self):
        \"\"\"
        Returns the test DataLoader for MNIST.
        For simplicity, we're reusing the validation dataloader.
        \"\"\"
        return self.val_dataloader()
"""
        with open("./custom_mnist.py", "w") as f:
            f.write(custom_datamodule)

        print("Also created custom_mnist.py with an ExtendedMNISTDataModule class.")
        print("If modifying the original module doesn't work, you can use this alternative.")
        print("Usage: from custom_mnist import ExtendedMNISTDataModule")

    return True


def fix_test_script():
    """
    Modify the test script to handle the case where test_dataloader is not available.
    """
    # Find the test script
    test_script = find_file("test_perceiver_local.py")
    if not test_script:
        print("Error: Could not find test_perceiver_local.py file.")
        return False

    print(f"Found test script at: {test_script}")

    # Create backup
    backup_file = str(test_script) + ".bak"
    import shutil

    shutil.copy2(test_script, backup_file)
    print(f"Created backup of test script at: {backup_file}")

    # Read the file
    with open(test_script, "r") as f:
        content = f.read()

    # Find the call to trainer.test
    test_pattern = r"test_result = trainer.test\(lit_model, datamodule=data\)"

    # Replace with a try-except block
    new_test_code = """try:
        # Try to run test with the datamodule
        test_result = trainer.test(lit_model, datamodule=data)
    except Exception as e:
        if "test_dataloader" in str(e):
            logger.warning("Test dataloader not available. Using validation dataloader for testing.")
            # Use the validation dataloader for testing
            test_result = trainer.test(lit_model, val_dataloaders=trainer.val_dataloaders)
        else:
            # Re-raise other exceptions
            raise"""

    # Replace in the content
    new_content = re.sub(test_pattern, new_test_code, content)

    # Write the modified content
    with open(test_script, "w") as f:
        f.write(new_content)

    print(f"Successfully modified test script to handle missing test_dataloader")

    return True


def find_file(filename, start_dir=None):
    """
    Find a file in the project directory structure.
    """
    if start_dir is None:
        start_dir = os.getcwd()

    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)

    return None


def main():
    print("=" * 80)
    print("MNIST DATAMODULE FIX - ADDING TEST_DATALOADER")
    print("=" * 80)

    # Try to fix the MNISTDataModule
    mnist_fixed = fix_mnist_datamodule()

    # Also fix the test script as a fallback
    test_script_fixed = fix_test_script()

    if mnist_fixed or test_script_fixed:
        print("\nFixes applied successfully!")

        if mnist_fixed:
            print("\n1. Added test_dataloader method to MNISTDataModule")

        if test_script_fixed:
            print("\n2. Modified test script to handle missing test_dataloader as a fallback")

        print("\nTo test if the issues are resolved, run your original script:")
        print("    python test_perceiver_local.py")
    else:
        print("\nFailed to apply fixes.")

    print("=" * 80)


if __name__ == "__main__":
    main()
