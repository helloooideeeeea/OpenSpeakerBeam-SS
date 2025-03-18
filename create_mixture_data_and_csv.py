import os
import random
import argparse
import pandas as pd
import torchaudio
import torch
import numpy as np

# 固定セグメント長（10秒 = 16000 * 10）
SEGMENT_LENGTH = 16000 * 10


def get_random_segment(waveform: torch.Tensor, seg_length: int = SEGMENT_LENGTH) -> torch.Tensor:
    """
    入力 waveform (1, T) から、ランダムに seg_length サンプル分を抽出する。
    waveform が短い場合はゼロパディングする。
    """
    _, T = waveform.shape
    if T >= seg_length:
        start = random.randint(0, T - seg_length)
        segment = waveform[:, start:start + seg_length]
    else:
        # waveform が短い場合は、右側にゼロパディング
        pad = seg_length - T
        segment = torch.nn.functional.pad(waveform, (0, pad))
    return segment


def scale_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    クリーン信号とノイズ信号に対し、指定された SNR (dB) となるようにノイズをスケールする。
    SNR = 10 * log10( P_clean / P_noise )
    """
    # 信号パワー（平均二乗）
    power_clean = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)
    # 目標ノイズパワー
    target_noise_power = power_clean / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / (power_noise + 1e-8))
    return noise * scaling_factor


def mix_signals(target: np.ndarray, interference: np.ndarray, noise: np.ndarray,
                sir_db: float, snr_db: float) -> np.ndarray:
    """
    target, interference, noise は numpy 配列 (T,) であるとする。
    - SIR: Signal-to-Interference Ratio (target vs interference)
    - SNR: Signal-to-Noise Ratio (target vs noise)
    各信号のパワーに応じて interference と noise をスケールし、合成混合信号を生成する。
    """
    # スケール interference で SIR を満たす
    power_target = np.mean(target ** 2)
    power_interference = np.mean(interference ** 2)
    # 目標 interference のパワー
    target_interference_power = power_target / (10 ** (sir_db / 10))
    scaling_factor_interference = np.sqrt(target_interference_power / (power_interference + 1e-8))
    interference_scaled = interference * scaling_factor_interference

    # スケール noise で SNR を満たす
    noise_scaled = scale_to_snr(target, noise, snr_db)

    mixture = target + interference_scaled + noise_scaled
    return mixture


def get_all_flac_files(librispeech_root: str) -> dict:
    """
    LibriSpeech のルートディレクトリ（例：data/train/LibriSpeech/train-clean）を走査して、
    各話者ごとに全ての FLAC ファイルのパスをリスト化した辞書を返す。
    キーは話者ID（上位ディレクトリ名）、値は各ファイルの絶対パスのリスト。
    """
    speakers = {}
    for speaker in os.listdir(librispeech_root):
        speaker_path = os.path.join(librispeech_root, speaker)
        if os.path.isdir(speaker_path):
            file_list = []
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                if os.path.isdir(chapter_path):
                    for fname in os.listdir(chapter_path):
                        if fname.endswith(".flac"):
                            file_list.append(os.path.join(chapter_path, fname))
            if file_list:
                speakers[speaker] = file_list
    return speakers


def get_noise_files(noise_root: str) -> list:
    """
    noise_root 内の全ての音声ファイルのパスリストを返す。
    """
    noise_files = []
    for fname in os.listdir(noise_root):
        if fname.endswith(".wav"):
            noise_files.append(os.path.join(noise_root, fname))
    return noise_files


def create_mixture_data_and_csv(args):
    """
    LibriSpeech と DNS4 のノイズを使って、シミュレーションした混合音声の CSV を作成する。
    CSV は、mixture_path, enrollment_path, target_path のカラムを持つ。
    生成するファイルは、固定長（10秒）のセグメントとする。
    """
    # 入力ディレクトリ
    libri_root = os.path.join(args.data_dir, "LibriSpeech", "train-clean")
    noise_root = os.path.join(args.data_dir, "noise_fullband")

    # 出力ディレクトリ（混合音声、enrollment、target を保存）
    mixture_dir = os.path.join(args.output_dir, "mixtures")
    enrollment_dir = os.path.join(args.output_dir, "enrollment")
    target_dir = os.path.join(args.output_dir, "target")
    os.makedirs(mixture_dir, exist_ok=True)
    os.makedirs(enrollment_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    # 生成するミックス数（例：50,000）
    num_mixtures = args.num_mixtures

    # SNR, SIR の範囲（dB）
    snr_range = (args.snr_min, args.snr_max)  # 例： (0, 25) for training
    sir_range = (args.sir_min, args.sir_max)  # 例： (-5, 5)

    # LibriSpeech の全 speaker の FLAC ファイル一覧を取得
    speakers = get_all_flac_files(libri_root)
    speaker_ids = list(speakers.keys())
    if len(speaker_ids) < 2:
        raise ValueError("LibriSpeech 内の話者が2人以上必要です。")

    # DNS4 のノイズファイル一覧
    noise_files = get_noise_files(noise_root)
    if len(noise_files) == 0:
        raise ValueError("ノイズファイルが見つかりません。")

    rows = []
    for i in range(num_mixtures):
        # ランダムにターゲット話者と干渉話者を選択（重複しないように）
        target_spk, interferer_spk = random.sample(speaker_ids, 2)

        # ターゲット話者から、混合用と enrollment 用に別々のファイルを選ぶ（できれば異なるファイル）
        target_files = speakers[target_spk]
        if len(target_files) < 2:
            continue  # もし十分な発話がない場合はスキップ
        target_mix_file, enrollment_file = random.sample(target_files, 2)

        # 干渉話者から混合用ファイルを選ぶ
        interferer_files = speakers[interferer_spk]
        if len(interferer_files) == 0:
            continue
        interferer_file = random.choice(interferer_files)

        # ロードして固定長セグメントを抽出
        try:
            target_waveform, sr = torchaudio.load(target_mix_file)
            interferer_waveform, _ = torchaudio.load(interferer_file)
            enrollment_waveform, _ = torchaudio.load(enrollment_file)
        except Exception as e:
            print(f"Error loading files: {e}")
            continue

        # サンプルレートが想定（16kHz）でない場合はリサンプリング
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            target_waveform = resampler(target_waveform)
            interferer_waveform = resampler(interferer_waveform)
            enrollment_waveform = resampler(enrollment_waveform)

        target_seg = get_random_segment(target_waveform)
        interferer_seg = get_random_segment(interferer_waveform)
        enrollment_seg = get_random_segment(enrollment_waveform)

        # ランダムに SNR, SIR を設定
        snr_db = random.uniform(*snr_range)
        sir_db = random.uniform(*sir_range)

        # ノイズファイルからランダムに選んでセグメント抽出
        noise_file = random.choice(noise_files)
        try:
            noise_waveform, noise_sr = torchaudio.load(noise_file)
        except Exception as e:
            print(f"Error loading noise file: {e}")
            continue
        if noise_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=16000)
            noise_waveform = resampler(noise_waveform)
        noise_seg = get_random_segment(noise_waveform)

        # 各セグメントを numpy 変換（1, T -> (T,)）
        target_np = target_seg.squeeze(0).numpy()
        interferer_np = interferer_seg.squeeze(0).numpy()
        noise_np = noise_seg.squeeze(0).numpy()

        # 混合信号を生成（まず target と interferer の比率を SIR で調整し、その後ノイズを SNR で加える）
        mixed_np = mix_signals(target_np, interferer_np, noise_np, sir_db, snr_db)

        # 保存先パスを決定
        mix_fname = f"mixture_{i:06d}.wav"
        enroll_fname = f"enrollment_{i:06d}.wav"
        target_fname = f"target_{i:06d}.wav"
        mix_path = os.path.join(mixture_dir, mix_fname)
        enroll_path = os.path.join(enrollment_dir, enroll_fname)
        target_path = os.path.join(target_dir, target_fname)

        # 保存（16kHz, 単一チャンネル）
        torchaudio.save(mix_path, torch.from_numpy(mixed_np).unsqueeze(0), 16000)
        torchaudio.save(enroll_path, enrollment_seg, 16000)  # enrollment はクリーンなセグメント
        torchaudio.save(target_path, target_seg, 16000)

        # CSV に記録
        rows.append({
            "mixture_path": mix_path,
            "enrollment_path": enroll_path,
            "target_path": target_path
        })

        if (i + 1) % 100 == 0:
            print(f"{i + 1} mixtures generated.")

    # CSV ファイルとして保存
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create CSV file and Mixture data for training data")
    parser.add_argument("--data_dir", type=str, default="data/train",
                        help="Root directory of training data (containing LibriSpeech and noise_fullband)")
    parser.add_argument("--output_dir", type=str, default="data_csv/train",
                        help="Output directory to save generated mixtures and CSV file")
    parser.add_argument("--num_mixtures", type=int, default=50000,
                        help="Number of mixtures to generate")
    parser.add_argument("--snr_min", type=float, default=0, help="Minimum SNR (dB)")
    parser.add_argument("--snr_max", type=float, default=25, help="Maximum SNR (dB)")
    parser.add_argument("--sir_min", type=float, default=-5, help="Minimum SIR (dB)")
    parser.add_argument("--sir_max", type=float, default=5, help="Maximum SIR (dB)")
    args = parser.parse_args()

    create_mixture_data_and_csv(args)
