# Perceiver, Perceiver IO and Perceiver AR

This repository is a PyTorch implementation of Perceiver, Perceiver IO and Perceiver AR, with PyTorch Lightning
interfaces for model training and Hugging Face 🤗 interfaces for inference.

<table>
  <tr>
    <td>
       <b>Perceiver</b>: General Perception with Iterative Attention
       (<a href="https://arxiv.org/abs/2103.03206">paper</a>,
        <a href="https://www.youtube.com/watch?v=P_xeshTnPZg">video</a>)
    </td>
    <td><img src="docs/images/small-perceiver.png" alt="Perceiver"/></td>
  </tr>
  <tr>
    <td>
      <b>Perceiver IO</b>: A General Architecture for Structured Inputs & Outputs
      (<a href="https://arxiv.org/abs/2107.14795">paper</a>,
       <a href="https://www.deepmind.com/blog/building-architectures-that-can-handle-the-worlds-data">blog post</a>)
    </td>
    <td><img src="docs/images/small-perceiver-io.png" alt="Perceiver IO"/></td>
  </tr>
  <tr>
    <td>
      General-purpose, long-context autoregressive modeling with <b>Perceiver AR</b>
      (<a href="https://arxiv.org/abs/2202.07765">paper</a>,
       <a href="https://www.deepmind.com/blog/perceiver-ar-general-purpose-long-context-autoregressive-generation">blog post</a>)
    </td>
    <td><img src="docs/images/small-perceiver-ar.png" alt="Perceiver AR"/></td>
  </tr>
</table>


## Comprehensive Implementation Check for Perceiver Architecture

`perceiver_pytorch.py`: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
### Core Architecture Components

| Component             | Paper Description                                   | Implementation in `perceiver_pytorch.py`                                  | Match? |
| --------------------- | --------------------------------------------------- | ------------------------------------------------------------------------- | ------ |
| **Overall Structure** | Cross-attention → Self-attention iteratively        | Implements modular cross-attention and self-attention blocks as described | ✓      |
| **Latent Array**      | Learned parameter array                             | `self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))`       | ✓      |
| **Cross-Attention**   | Asymmetric attention with queries from latent array | `Attention` class with query from latent, key/value from input            | ✓      |
| **Self-Attention**    | Standard Transformer-style self-attention           | `Attention` class with matching implementation                            | ✓      |
| **Feed-Forward**      | Transformer-style with expansion                    | `FeedForward` with expansion factor (`mult`) of 4                         | ✓      |

### Specific Numeric Values

|Parameter|Value in Paper|Implementation Default|Match?|
|---|---|---|---|
|`num_latents`|512 (ImageNet)|512|✓|
|`latent_dim`|1024 (ImageNet)|512|⚠️ Different|
|`cross_heads`|1|1|✓|
|`latent_heads`|8|8|✓|
|`cross_dim_head`|Not explicitly stated|64|N/A|
|`latent_dim_head`|Not explicitly stated|64|N/A|
|`self_per_cross_attn`|6 (ImageNet)|1|⚠️ Different|
|`num_freq_bands`|64 (ImageNet)|Required parameter|N/A|
|`depth`|8 iterations × 6 blocks = 48|Required parameter|N/A|

### Implementation Details

|Feature|Paper Description|Implementation|Match?|
|---|---|---|---|
|**Activation Function**|GELU|`GEGLU()` using F.gelu|✓|
|**Layer Normalization**|Before attention and FF|`PreNorm` wrapper around both|✓|
|**Attention Formula**|Standard scaled dot-product attention|Correctly implemented with scale factor|✓|
|**Feed-Forward**|MLP with GELU, no bottleneck|Uses GEGLU with expansion factor 4|✓|
|**Weight Tying**|Shared weights except first cross-attend|`cache_fn` decorator with `_cache` control|✓|
|**Position Encoding**|Fourier features with concatenation|`fourier_encode` function with correct math|✓|
|**Output Pooling**|Global average over latents|`Reduce('b n d -> b d', 'mean')`|✓|
|**Output Classification**|Linear layer after pooling|Linear layer after normalization|✓|

### Attention Implementation

|Detail|Paper Description|Implementation|Match?|
|---|---|---|---|
|**QKV Projection**|Linear projections|Separate linear layers for Q, K, V|✓|
|**Multi-head Implementation**|Reshape into multiple heads|Correct reshaping with `rearrange`|✓|
|**Attention Scale**|1/√(dimension per head)|`self.scale = dim_head ** -0.5`|✓|
|**Mask Handling**|Optional masking|Mask handling with correct values|✓|

### Fourier Position Encoding

|Detail|Paper Description|Implementation|Match?|
|---|---|---|---|
|**Frequency Generation**|Linearly spaced up to max_freq/2|`torch.linspace(1., max_freq / 2, num_bands)`|✓|
|**Encoding Function**|sin/cos pairs + original position|`torch.cat([x.sin(), x.cos()], dim = -1)` + concatenation with original|✓|
|**Position Scaling**|Multiplied by π|`x = x * scales * pi`|✓|

### Input/Output Processing

|Detail|Paper Description|Implementation|Match?|
|---|---|---|---|
|**Input Reshaping**|Flattening spatial dimensions|`rearrange(data, 'b ... d -> b (...) d')`|✓|
|**Position Handling**|Position encoding concatenated|Concatenated to input data|✓|
|**Final Classification**|Global pooling + normalization|Global pooling + layer norm + linear|✓|

### Key Implementation Differences

1. **Default latent dimension**: Paper uses 1024 for ImageNet while implementation defaults to 512
2. **Self-attention blocks**: Paper uses 6 per cross-attention while implementation defaults to 1
3. **Weight sharing**: Paper shares weights by default for all but first cross-attention, implementation defaults to no sharing

The implementation is structurally correct and follows the paper's architecture, but users would need to adjust parameters from defaults to match the exact configurations used in the original paper's experiments.

## Overview

Core of the `perceiver-io` library are *backend models*, lightweight PyTorch implementations of Perceiver,
Perceiver IO and Perceiver AR. They can be wrapped into [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
modules for training (*Lightning interface*) and 🤗 modules for inference (*Hugging Face interface*). See
[library design](docs/library-design.md) for details.

<p align="center">
    <img src="docs/images/library-design-small.jpg" alt="library-design"/>
</p>

The command line interface for training is implemented with [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html).
Training datasets are 🤗 [datasets](https://huggingface.co/docs/datasets) wrapped into PyTorch Lightning data modules.
For NLP tasks, `perceiver-io` supports all 🤗 [fast tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers)
and the 🤗 Perceiver UTF-8 bytes tokenizer.

## Documentation

- [Installation](#installation)
- [Getting started](#getting-started)
- [Library design](docs/library-design.md)
- [Pretrained models](docs/pretrained-models.md)
- [Training examples](docs/training-examples.md)
- [Inference examples](examples/inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/perceiver-io/blob/main/examples/inference.ipynb)
- [Model construction](docs/model-construction.md)
- [Building blocks](docs/building-blocks.md)

## Installation

### Via pip

```shell
pip install perceiver-io[text,vision,audio]
```

### From sources

Installation from sources requires a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and a
[Poetry](https://python-poetry.org/docs/#installation) (1.2.0 or higher) installation.

Create and activate the `perceiver-io` conda environment:

```shell
conda env create -f environment.yml
conda activate perceiver-io
```

Install main and test dependencies, including all extras:

```shell
# Without dependencies required for examples
poetry install --all-extras
```

If you want to run the [examples](examples) locally, additionally use `--with examples`:

```shell
poetry install --all-extras --with examples
```

### Docker image

```shell
docker pull ghcr.io/krasserm/perceiver-io:latest
```

See [Docker image](docs/docker-image.md) for details.

## Getting started

### Inference

#### Optical flow

Compute the optical flow between consecutive frames of an input video and write the rendered results to an output
video:

```python
from urllib.request import urlretrieve
from transformers import pipeline

from perceiver.data.vision import video_utils
from perceiver.model.vision import optical_flow  # register auto-classes and pipeline

urlretrieve(
    url="https://martin-krasser.com/perceiver/flow/sintel_clip_cave_dragon_fight.mp4",
    filename="sintel_clip_cave_dragon_fight.mp4",
)

# Create optical flow pipeline
optical_flow_pipeline = pipeline("optical-flow", model="krasserm/perceiver-io-optical-flow", device="cuda:0")

# load consecutive video frame pairs
frame_pairs = video_utils.read_video_frame_pairs("sintel_clip_cave_dragon_fight.mp4")

# create and render optical flow for all frame pairs
optical_flows = optical_flow_pipeline(frame_pairs, render=True, device="cuda:0")

# create video with rendered optical flows
video_utils.write_video("sintel_clip_cave_dragon_fight_output.mp4", optical_flows, fps=24)
```

Here is a side-by-side comparison of the input and output video:

<p align="center">
    <img src="docs/images/optical-flow.gif" alt="optical-flow-sbs">
</p>

#### Symbolic audio generation

Create audio sequences by generating symbolic ([MIDI](https://en.wikipedia.org/wiki/MIDI)) audio data and converting the
generated audio symbols into WAV output using [fluidsynth](https://www.fluidsynth.org/) (_Note:_ fluidsynth must be installed
in order for the following example to work):  

```python
from transformers import pipeline
from pretty_midi import PrettyMIDI
from perceiver.model.audio import symbolic  # auto-class registration

repo_id = "krasserm/perceiver-ar-sam-giant-midi"

prompt = PrettyMIDI("prompt.mid")
audio_generator = pipeline("symbolic-audio-generation", model=repo_id)

output = audio_generator(prompt, max_new_tokens=64, num_latents=1, do_sample=True, top_p=0.95, temperature=1.0, render=True)

with open("generated_audio.wav", "wb") as f:
    f.write(output["generated_audio_wav"])
```

Examples of generated audio sequences are available on the 🤗 [hub](https://huggingface.co/krasserm/perceiver-ar-sam-giant-midi#audio-samples).

See [inference examples](https://colab.research.google.com/github/krasserm/perceiver-io/blob/main/examples/inference.ipynb)
for more examples.

### Training

Train a small Perceiver IO image classifier (907K parameters) on MNIST from the command line. The classifier
cross-attends to individual pixels of input images with [repeated cross-attention](docs/building-blocks.md).
See [image classification](docs/training-examples.md#image-classification) training example for more details.

```shell
python -m perceiver.scripts.vision.image_classifier fit \
  --model.num_latents=32 \
  --model.num_latent_channels=128 \
  --model.encoder.num_frequency_bands=32 \
  --model.encoder.num_cross_attention_layers=2 \
  --model.encoder.num_self_attention_blocks=3 \
  --model.encoder.num_self_attention_layers_per_block=3 \
  --model.encoder.first_self_attention_block_shared=false \
  --model.encoder.dropout=0.1 \
  --model.encoder.init_scale=0.1 \
  --model.decoder.num_output_query_channels=128 \
  --model.decoder.dropout=0.1 \
  --model.decoder.init_scale=0.1 \
  --data=MNISTDataModule \
  --data.batch_size=64 \
  --optimizer=AdamW \
  --optimizer.lr=1e-3 \
  --lr_scheduler.warmup_steps=500 \
  --trainer.accelerator=gpu \
  --trainer.devices=1 \
  --trainer.max_epochs=30 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=logs
```

[Model construction](docs/model-construction.md) describes how to implement model-specific command line interfaces
with the Lightning CLI. Training checkpoints are written to the `logs/img_clf/version_0/checkpoints` directory. Assuming
a checkpoint with filename `epoch=025-val_loss=0.065.ckpt` exists, it can be converted to a `perceiver-io` 🤗 model with

```python
from perceiver.model.vision.image_classifier import convert_mnist_classifier_checkpoint

convert_mnist_classifier_checkpoint(
    save_dir="example/mnist-classifier",
    ckpt_url="logs/img_clf/version_0/checkpoints/epoch=025-val_loss=0.065.ckpt",
)
```

so that it can be used in a 🤗 image classification pipeline

```python
from datasets import load_dataset
from transformers import pipeline

mnist_dataset = load_dataset("mnist", split="test")[:9]

images = mnist_dataset["image"]
labels = mnist_dataset["label"]

classifier = pipeline("image-classification", model="example/mnist-classifier")
predictions = [pred[0]["label"] for pred in classifier(images)]

print(f"Labels:      {labels}")
print(f"Predictions: {predictions}")
```
```
Labels:      [7, 2, 1, 0, 4, 1, 4, 9, 5]
Predictions: [7, 2, 1, 0, 4, 1, 4, 9, 5]
```

or loaded directly:

```python
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

model = AutoModelForImageClassification.from_pretrained("example/mnist-classifier")
processor = AutoImageProcessor.from_pretrained("example/mnist-classifier")

inputs = processor(images, return_tensors="pt")

with torch.no_grad():
    # use perceiver-io Hugging Face model
    output_1 = model(**inputs).logits

with torch.no_grad():
    # or use perceiver-io backend model directly  
    output_2 = model.backend_model(inputs.pixel_values)

print(f"Predictions: {output_1.argmax(dim=-1).numpy().tolist()}")
print(f"Predictions: {output_2.argmax(dim=-1).numpy().tolist()}")
```
```
Predictions: [7, 2, 1, 0, 4, 1, 4, 9, 5]
Predictions: [7, 2, 1, 0, 4, 1, 4, 9, 5]
```

See [training examples](docs/training-examples.md) for more examples.

## Articles

Articles referencing this repository:

- [Training compute-optimal Perceiver AR language models](https://krasserm.github.io/2023/01/23/scaling-perceiver-ar/)
- [A gentle introduction to Rotary Position Embedding](https://krasserm.github.io/2022/12/13/rotary-position-embedding/)

## Other implementations

- [Perceiver](https://paperswithcode.com/paper/perceiver-general-perception-with-iterative#code)
- [Perceiver IO](https://paperswithcode.com/paper/perceiver-io-a-general-architecture-for#code)
- [Perceiver AR](https://paperswithcode.com/paper/general-purpose-long-context-autoregressive#code)
