import torch
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

def get_speaker_embeddings_batch(speaker_encoder: VoiceEncoder, enrollment_batch: torch.Tensor) -> torch.Tensor:
    """
    enrollment 音声のバッチから、各サンプルの d-vector エンベディングを生成する関数。

    Args:
        speaker_encoder (VoiceEncoder): d-vector を生成するための音声エンコーダ。
        enrollment_batch (torch.Tensor): 入力 enrollment 音声テンソル、形状は (B, 1, T)。
            ※ 各サンプルは、ターゲットサンプルレートであることを仮定。

    Returns:
        torch.Tensor: 各 enrollment に対応する d-vector エンベディングテンソル、形状は (B, embedding_dim)。
    """
    enroll_list = []
    B = enrollment_batch.size(0)
    for i in range(B):
        waveform = enrollment_batch[i]  # shape: (1, T)
        # enrollment のサンプルレートが target_sr であると仮定（必要ならリサンプリングを実施）
        waveform_np = waveform.squeeze(0).cpu().numpy()  # (T,)
        # 前処理: VoiceEncoder 用に加工
        processed_wav = preprocess_wav(waveform_np)
        enroll_list.append(processed_wav)

    # 各 enrollment に対して embed_utterance を使い、個別の d-vector を計算
    embeddings = []
    for wav in enroll_list:
        emb = speaker_encoder.embed_utterance(wav)
        embeddings.append(emb)
    # embeddings_np の形状は (B, embedding_dim) となる
    embeddings_np = np.stack(embeddings, axis=0)
    embeddings_tensor = torch.from_numpy(embeddings_np).to(enrollment_batch.device)
    return embeddings_tensor


if __name__ == "__main__":
    # enrollment 音声テンソル (バッチサイズ2, チャンネル1, サンプル数)
    waveform1 = torch.randn(1, 203776)
    waveform2 = torch.randn(1, 203776)
    batch_waveform = torch.stack([waveform1, waveform2], dim=0)
    print("batch_waveform shape:", batch_waveform.shape)  # (2, 1, 203776)

    speaker_encoder = VoiceEncoder(device="cpu")
    speaker_embeddings = get_speaker_embeddings_batch(speaker_encoder, batch_waveform)
    print("speaker_embeddings shape:", speaker_embeddings.shape)
    # (2, 256)