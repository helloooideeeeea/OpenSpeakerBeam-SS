# SpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling

This is an **ongoing implementation** of [SpeakerBeam-SS](https://arxiv.org/abs/2407.01857), a real-time target speaker extraction model combining lightweight Conv-TasNet and State Space Modeling (S4D). The goal is to achieve efficient and high-performance speaker separation on resource-constrained devices.

🚨 **Disclaimer:** This repository is **not affiliated** with the authors of the original paper. It is an independent reimplementation and may have differences from the paper's methodology. If you have suggestions for improvements, feel free to share them! 🚨

## 🚧 Work in Progress 🚧

This repository is under active development. The **network model implementation is complete**, and the next step is **generating test data**. Some architectural differences from the original paper may exist, and feedback is welcome.

## 📖 Reference

- **Paper:** [SpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling](https://arxiv.org/abs/2407.01857)

## 📌 Features

- Conv-TasNet-based architecture with **S4D blocks** for efficient temporal modeling
- **Multiplicative adaptation** with d-vector speaker embeddings
- **1D convolutional blocks** for feature extraction
- **ONNX Runtime support** for CPU acceleration (AVX2 / AVX-512)
- **Designed for real-time inference** on mobile and server environments

## 🔧 Installation

### Dependencies

Install required dependencies with:

```sh
pip install -r requirements.txt
```

## 🚀 Usage

```python
import torch
from model.speakerbeam import SpeakerBeamSS

# Dummy input (1-second speech at 16kHz)
batch_size = 1
input_len = 16000  # 1 sec @ 16kHz
mixture = torch.randn(batch_size, 1, input_len)
speaker_embedding = torch.randn(1, 256)  # d-vector use Resemblyzer

# Load model
model = SpeakerBeamSS()

# Run inference
with torch.no_grad():
    output = model(mixture, speaker_embedding)

print("Input Shape:", mixture.shape)
print("Output Shape:", output.shape)
```

## 💡 Performance

Initial FLOP measurements on 1-second input (16kHz):

```
FLOPs: 21.60G, Params: 7.64M
```

- Expected to run **in real-time on modern CPUs** with **AVX2 or AVX-512** optimizations.
- **Neon acceleration** planned for **iOS devices** via ONNX Runtime.

## 📌 TODO

- Generate test datasets
- Validate output quality
- Optimize model for mobile deployment

## 📜 License

TBD (likely MIT or Apache 2.0)

## 🙌 Acknowledgments

This work is inspired by the original SpeakerBeam-SS paper and the Conv-TasNet framework.

🔹 **Speaker embeddings are generated using [Resemblyzer](https://github.com/resemble-ai/Resemblyzer/).**

