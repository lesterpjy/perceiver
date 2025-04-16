import os
from typing import Optional, Tuple, List
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from perceiver.data.vision.common import channels_to_last, ImagePreprocessor, lift_transform


class CIFAR100Preprocessor(ImagePreprocessor):
    def __init__(self, normalize: bool = True, channels_last: bool = True):
        super().__init__(cifar100_transform(normalize, channels_last))


def cifar_collate_fn(batch):
    """Custom collate function to ensure proper tensor conversion for CIFAR100 dataset."""
    images = []
    labels = []

    for item in batch:
        if isinstance(item["image"], torch.Tensor):
            images.append(item["image"])
        else:
            # Convert PIL
            to_tensor = transforms.ToTensor()
            images.append(to_tensor(item["image"]))

        labels.append(item["fine_label"])

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return {"image": images, "label": labels}


def lift_cifar_transform(transform):
    """Simple transform lifter for CIFAR100."""
    def apply(examples):
        # CIFAR100 dataset uses "img" as the key for images
        examples["image"] = [transform(img) for img in examples["img"]]
        return examples
    return apply


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = os.path.join(".cache", "cifar100"),
        normalize: bool = True,
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.channels_last = channels_last

        self.tf_train = cifar100_transform(normalize, channels_last, random_crop=random_crop)
        self.tf_valid = cifar100_transform(normalize, channels_last, random_crop=None)

        self.ds_train = None
        self.ds_valid = None

    @property
    def num_classes(self):
        return 100

    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return 32, 32, 3
        else:
            return 3, 32, 32

    def load_dataset(self, split: Optional[str] = None):
        return load_dataset("cifar100", split=split, cache_dir=self.hparams.dataset_dir)

    def prepare_data(self) -> None:
        self.load_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = self.load_dataset(split="train")
        self.ds_train.set_transform(lift_cifar_transform(self.tf_train))

        self.ds_valid = self.load_dataset(split="test")
        self.ds_valid.set_transform(lift_cifar_transform(self.tf_valid))

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            shuffle=self.hparams.shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=cifar_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=cifar_collate_fn,
        )

    def test_dataloader(self):
        """
        Returns the test DataLoader for CIFAR100.
        For simplicity, we're reusing the validation dataset.
        """
        return DataLoader(
            self.ds_valid,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=cifar_collate_fn,
        )


def cifar100_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    # Data augmentation for training
    if random_crop is not None:
        transform_list.append(transforms.RandomCrop(32, padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)