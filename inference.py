import os
import torch
import torchaudio
import argparse
from model import SpeakerBeamSS
from resemblyzer import VoiceEncoder
from tools import get_speaker_embeddings_batch

def main():
    parser = argparse.ArgumentParser(description="Inference for SpeakerBeam-SS")
    parser.add_argument("--mixture", type=str, required=True,
                        help="Path to the input mixture audio file")
    parser.add_argument("--enrollment", type=str, required=True,
                        help="Path to the enrollment audio file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output (enhanced) wav file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing the best_model.pth checkpoint")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- 1. 混合音声の読み込みと前処理 -----
    mixture_waveform, sr_m = torchaudio.load(args.mixture)
    if sr_m != args.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_m, new_freq=args.sample_rate)
        mixture_waveform = resampler(mixture_waveform)
    # モノラル化（必要なら平均化）
    if mixture_waveform.shape[0] > 1:
        mixture_waveform = torch.mean(mixture_waveform, dim=0, keepdim=True)
    # バッチ次元追加（形状: (B, 1, T) ）
    if mixture_waveform.dim() == 2:
        mixture_waveform = mixture_waveform.unsqueeze(0)
    mixture_waveform = mixture_waveform.to(device)

    # ----- 2. enrollment 音声の読み込みと前処理 -----
    enrollment_waveform, sr_e = torchaudio.load(args.enrollment)
    if sr_e != args.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_e, new_freq=args.sample_rate)
        enrollment_waveform = resampler(enrollment_waveform)
    if enrollment_waveform.shape[0] > 1:
        enrollment_waveform = torch.mean(enrollment_waveform, dim=0, keepdim=True)
    if enrollment_waveform.dim() == 2:
        enrollment_waveform = enrollment_waveform.unsqueeze(0)
    enrollment_waveform = enrollment_waveform.to(device)

    # ----- 3. 学習済みモデルと Speaker Encoder のロード -----
    model = SpeakerBeamSS().to(device)
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found at {best_model_path}. Train the model first.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # SpeakerEncoder を初期化
    speaker_encoder = VoiceEncoder(device=device)

    # ----- 4. enrollment 音声からスピーカーエンベディングを取得 -----
    speaker_embeddings = get_speaker_embeddings_batch(speaker_encoder, enrollment_waveform)
    # 期待する形状: (B, embedding_dim)

    # ----- 5. 推論実行 -----
    with torch.no_grad():
        enhanced = model(mixture_waveform, speaker_embeddings)
    # enhanced の形状: (B, 1, T)

    # ----- 6. 推論結果を wav として保存 -----
    enhanced = enhanced.cpu()
    torchaudio.save(args.output, enhanced, args.sample_rate)
    print(f"Inference complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
