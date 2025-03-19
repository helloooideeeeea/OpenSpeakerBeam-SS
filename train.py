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

def fix_length(waveform: torch.Tensor, target_length: int = 7*16000) -> torch.Tensor:
    """
    入力の waveform (shape: (C, T)) を target_length サンプルに固定する関数。
    T > target_length の場合は先頭 target_length サンプルを抽出し、
    T < target_length の場合は末尾にゼロパディングする。
    """
    current_length = waveform.size(1)
    if current_length > target_length:
        return waveform[:, :target_length]
    elif current_length < target_length:
        pad_length = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, pad_length))
    else:
        return waveform

def enrollment_fix_length(waveform: torch.Tensor) -> torch.Tensor:
    return fix_length(waveform, target_length=7 * 16000)

# ========================================
# 1. SI-SNR loss 関数
# ========================================
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


# ========================================
# 2. Dataset の定義
# ========================================
class SpeechDataset(Dataset):
    """
    CSVファイルに記載された音声パスから、mixture, enrollment, target のペアを返す Dataset
    CSV ファイルは、少なくとも以下のカラムを含むものとする:
        - mixture_path
        - enrollment_path
        - target_path
    """

    def __init__(self, csv_file, transform=None, enrollment_transform = None):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.enrollment_transform = enrollment_transform

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
            target = self.transform(target)
        if self.enrollment_transform:
            enrollment = self.enrollment_transform(enrollment)
        return mixture, enrollment, target


# ========================================
# 3. 検証 / テスト時用の評価関数
# ========================================
@torch.no_grad()
def evaluate(model, dataloader, speaker_encoder, device):
    """DevやTestでSI-SNRを計算する共通関数"""
    model.eval()
    total_loss = 0.0
    for mixture, enrollment, target in dataloader:
        mixture = mixture.to(device)
        enrollment = enrollment.to(device)
        target = target.to(device)

        speaker_embeddings = get_speaker_embeddings_batch(speaker_encoder, enrollment)
        output = model(mixture, speaker_embeddings)

        output = output.squeeze(1)
        target = target.squeeze(1)

        loss = si_snr_loss(target, output)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()  # ここで学習モードに戻す
    return avg_loss


# ========================================
# 4. メイン学習関数
# ========================================
def train_and_validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # (A) DataLoader の準備
    # ---------------------------
    # Trainデータ
    train_dataset = SpeechDataset(csv_file=args.train_csv, enrollment_transform=enrollment_fix_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Devデータ（ハイパーパラメータ調整・性能検証用）
    dev_dataset = SpeechDataset(csv_file=args.dev_csv, enrollment_transform=enrollment_fix_length)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 音声埋め込みエンコーダー
    speaker_encoder = VoiceEncoder(device=device)

    # ---------------------------
    # (B) モデルやオプティマイザの定義
    # ---------------------------
    model = SpeakerBeamSS().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ReduceLROnPlateau の設定
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.reduce_patience, verbose=True
    )

    # 学習開始
    model.train()
    global_step = 0

    best_dev_loss = float("inf")  # Devの最小損失を追跡
    patience_count = 0            # Early Stopping用カウンタ

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0

        # ---------------------------
        # (C) Trainエポック
        # ---------------------------
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
                print(f"[Train] Epoch {epoch+1}/{args.num_epochs}, "
                      f"Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # ---------------------------
        # (D) Devエポック (検証)
        # ---------------------------
        dev_loss = evaluate(model, dev_loader, speaker_encoder, device)
        print(f"[Dev]   Epoch {epoch+1}/{args.num_epochs}, Dev Loss: {dev_loss:.4f}")

        # スケジューラにDev損失を渡して学習率を調整（ReduceLROnPlateauなど）
        scheduler.step(dev_loss)

        # ---------------------------
        # (E) ベストモデルの更新 & 早期停止判定
        # ---------------------------
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_count = 0

            # ベストモデルを保存（Dev損失が改善したとき）
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"=> Best model updated! Dev Loss = {dev_loss:.4f}")
        else:
            # 改善しなかった場合
            patience_count += 1
            if patience_count >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

        print(f"[Train] Epoch {epoch+1} finished! Average Train Loss: {avg_train_loss:.4f}\n")

    # ---------------------------
    # 学習終了後、最終的にベストモデルをロードしておく
    # ---------------------------
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("No best model found (no improvement on Dev set).")

    return model


# ========================================
# 5. テスト時の評価関数
# ========================================
def test_model(args):
    """
    学習済み(ベスト)モデルを使ってテストデータで評価する関数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) テストデータローダーの用意
    test_dataset = SpeechDataset(csv_file=args.test_csv, enrollment_transform=enrollment_fix_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2) モデルとスピーカーエンコーダをロード
    speaker_encoder = VoiceEncoder(device=device)
    model = SpeakerBeamSS().to(device)

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Train the model first.")

    # 3) ベストモデルを読み込み
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path} for testing.")

    # 4) テストデータ上で評価 (SI-SNR)
    test_loss = evaluate(model, test_loader, speaker_encoder, device)
    print(f"[Test] Test Loss (SI-SNR): {test_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train & Validate SpeakerBeam-SS with SI-SNR loss")
    parser.add_argument("--train_csv", type=str, default="data_csv/train/metadata.csv")
    parser.add_argument("--dev_csv", type=str, default="data_csv/dev/metadata.csv")
    parser.add_argument("--test_csv", type=str, default="data_csv/test/metadata.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Early Stopping用パラメータ
    parser.add_argument("--early_stop_patience", type=int, default=120,
                        help="Number of epochs to wait for dev_loss improvement before early stopping.")
    # ReduceLROnPlateau用パラメータ
    parser.add_argument("--reduce_patience", type=int, default=20,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")

    parser.add_argument("--mode", type=str, default="train",
                        help="Specify 'train' or 'test'. If 'test', evaluate on test data.")

    args = parser.parse_args()

    if args.mode == "train":
        trained_model = train_and_validate(args)
    elif args.mode == "test":
        test_model(args)
    else:
        raise ValueError("--mode should be 'train' or 'test'.")
