OpenSpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling

This is an independent implementation of SpeakerBeam-SS, a real-time target speaker extraction model combining lightweight Conv-TasNet and State Space Modeling (S4D). The goal is to achieve efficient and high-performance speaker separation on resource-constrained devices.

🚨 Disclaimer: This repository is not affiliated with the authors of the original paper. It is an independent reimplementation and may have differences from the paper's methodology. If you have suggestions for improvements, feel free to share them! 🚨

✅ Project Status

The network model implementation, training, and test dataset preparation are complete. A full training cycle has been conducted using datasets published on Hugging Face, and test results are available. Some architectural differences from the original paper may exist. Feedback and pull requests are welcome.

📖 Reference

Paper: SpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling

📌 Features

Conv-TasNet-based architecture with S4D blocks for efficient temporal modeling

Multiplicative adaptation with d-vector speaker embeddings

1D convolutional blocks for feature extraction

ONNX Runtime support for CPU acceleration (AVX2 / AVX-512)

Designed for real-time inference on mobile and server environments

🔧 Installation

Dependencies

Install required dependencies with:

pip install -r requirements.txt

🚀 Usage

🔊 Inference

Run speaker extraction on a given mixture and enrollment audio:

python inference.py \
  --mixture data/sample/mixture_000001.wav \
  --enrollment data/sample/enrollment_000001.wav \
  --output data/sample/result_000001.wav

🏋️ Training

python train.py --mode=train

🧪 Testing

python train.py --mode=test

Training and testing CSV metadata files are automatically downloaded and stored from Hugging Face:

--train_csv data_csv/train/metadata.csv
--dev_csv   data_csv/dev/metadata.csv
--test_csv  data_csv/test/metadata.csv

💾 Dataset & Checkpoints

✅ Test dataset and pretrained model available on Hugging Face:https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset/tree/main

✅ Pretrained model: checkpoints/best_model.pth

🔍 [Test] Test Loss (SI-SNR): -5.8925(Note: current performance is modest; improvements are planned.)

Output samples will be saved as:

data/sample/enrollment_000001.wav  <audio>
data/sample/mixture_000001.wav     <audio>
data/sample/result_000001.wav      <audio>

data/sample/enrollment_000002.wav  <audio>
data/sample/mixture_000002.wav     <audio>
data/sample/result_000002.wav      <audio>

💡 Performance

Initial FLOP measurements on 1-second input (16kHz):

FLOPs: 21.60G, Params: 7.64M

Expected to run in real-time on modern CPUs with AVX2 or AVX-512 optimizations.

Neon acceleration planned for iOS devices via ONNX Runtime.

📌 TODO

Validate output quality

Optimize model for mobile deployment

📜 License

TBD (likely MIT or Apache 2.0)

🙌 Acknowledgments

This work is inspired by the original SpeakerBeam-SS paper and the Conv-TasNet framework.

🔹 Speaker embeddings are generated using Resemblyzer.

