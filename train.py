import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import SpeakerBeamSS
from tools import get_speaker_embeddings_batch
from resemblyzer import VoiceEncoder


# ---------------------------------------------
# SI-SNR loss 関数の定義
# ---------------------------------------------
def si_snr_loss(s, s_hat, eps=1e-8):
    """
    SI-SNR loss を計算する関数。

    Args:
        s (Tensor): 正解音声 (B, T)
        s_hat (Tensor): 推定音声 (B, T)
        eps (float): 数値安定性のための微小値
    Returns:
        loss (Tensor): 平均の負の SI-SNR (dB) 値
    """
    # 各サンプルごとに平均を引いてゼロ平均化
    s = s - torch.mean(s, dim=1, keepdim=True)
    s_hat = s_hat - torch.mean(s_hat, dim=1, keepdim=True)

    # 正解信号への射影（スケール不変）
    s_target = torch.sum(s_hat * s, dim=1, keepdim=True) / (torch.sum(s * s, dim=1, keepdim=True) + eps) * s
    e_noise = s_hat - s_target
    ratio = torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps)
    # SI-SNR [dB]
    si_snr = 10 * torch.log10(ratio + eps)

    # 損失は負の SI-SNR の平均（最大化が目的なので最小化問題に変換）
    loss = -torch.mean(si_snr)
    return loss


# ---------------------------------------------
# Dataset の定義（CSV に各音声パスが記載されていると仮定）
# ---------------------------------------------
class SpeechDataset(Dataset):
    """
    CSVファイルに記載された音声パスから、mixture, enrollment, target のペアを返す Dataset
    CSV ファイルは、少なくとも以下のカラムを含むものとする:
        - mixture_path
        - enrollment_path
        - target_path
    """

    def __init__(self, csv_file, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        mixture, sr1 = torchaudio.load(row["mixture_path"])
        enrollment, sr2 = torchaudio.load(row["enrollment_path"])
        target, sr3 = torchaudio.load(row["target_path"])

        # サンプルレート等がすでに16kHz、モノラル、32bit floatである前提
        # 必要に応じて前処理（例: リサンプリング、正規化）を実施
        if self.transform:
            mixture = self.transform(mixture)
            enrollment = self.transform(enrollment)
            target = self.transform(target)
        return mixture, enrollment, target


# ---------------------------------------------
# メイン学習ループ
# ---------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset, DataLoader の準備
    train_dataset = SpeechDataset(csv_file=args.train_csv)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 音声埋め込みエンコーダーのインスタンスを作成
    speaker_encoder = VoiceEncoder(device=device)

    # モデルのインスタンス化
    model = SpeakerBeamSS().to(device)

    # オプティマイザの設定
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    model.train()
    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch_idx, (mixture, enrollment, target) in enumerate(train_loader):
            # mixture, enrollment, target は形状が (B, 1, T)
            mixture = mixture.to(device)
            enrollment = enrollment.to(device)
            target = target.to(device)

            # enrollment 音声からスピーカーエンベディングを取得
            speaker_embeddings = get_speaker_embeddings_batch(speaker_encoder, enrollment)

            optimizer.zero_grad()
            # モデルの順伝播: mixture と speaker_embeddings を入力し、推定音声を出力
            output = model(mixture, speaker_embeddings)
            # 出力・target の形状: (B, 1, T) → SI-SNRは (B, T) で計算するため squeeze する
            output = output.squeeze(1)
            target = target.squeeze(1)

            loss = si_snr_loss(target, output)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] Average Loss: {avg_loss:.4f}")

        # scheduler に平均損失を渡して学習率を更新
        scheduler.step(avg_loss)

        # 定期的にチェックポイントを保存
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"model_epoch{epoch + 1}.pth")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SpeakerBeam-SS with SI-SNR loss")
    parser.add_argument("--train_csv", type=str, default="train_metadata.csv",
                        help="Path to CSV file containing training data paths")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=5, help="Checkpoint save interval (epochs)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    train(args)


