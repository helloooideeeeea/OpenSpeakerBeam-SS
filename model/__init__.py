import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.masknn.convolutional import Conv1DBlock  # Conv-TasNet style 1-D convolution block
from model.s4d import S4D  # S4D layer implementation
from model.adapt_layers import MulAddAdaptLayer  # Multiplicative adaptation (or FiLM) layer
from tools import get_speaker_embeddings_batch


# =====================
#  Encoder
# =====================
class Encoder(nn.Module):
    """
    1D Convolutional Encoder.

    This module maps the input waveform into a latent space.
    It applies a single Conv1d followed by a ReLU activation.

    Input shape: (batch, 1, time)
    Output shape: (batch, out_channels, time')
    """

    def __init__(self, kernel_size=320, stride=160, out_channels=4096):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,  # mono input
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        x = self.conv(x)  # Convolve to produce latent representation: (B, 4096, time')
        x = F.relu(x)  # Apply ReLU non-linearity
        return x


# ------------------------------------------------------------------------
# S4D Block
# ------------------------------------------------------------------------
class S4DBlock(nn.Module):
    """
    S4D Block as interpreted from Figure 1(b) of the paper.

    The block consists of:
      1) LayerNorm followed by S4D layer,
      2) A small feed-forward block: GELU activation followed by a 1x1 conv,
      3) A gated mechanism using GLU to merge the original input and the S4D output,
         with a skip connection: A = x + gated_output,
      4) A second feed-forward block (LN -> Linear -> GELU -> Linear),
         followed by a final residual addition: B = A + feed_forward_output.

    Note: The exact architecture is an interpretation. The "Linear" layers are implemented
          as 1x1 convolutions, which perform a per-time-frame linear transform on the channel dimension.
    """

    def __init__(self, d_model=256, d_state=32):
        super().__init__()

        # 1) Apply LayerNorm (channel-wise) before S4D
        self.ln_s4d = nn.LayerNorm(d_model)
        # S4D layer: models long-term dependencies using state-space modeling
        self.s4d = S4D(
            d_model=d_model,
            d_state=d_state,
            dropout=0.0,
            transposed=True  # works on (B, C, T)
        )
        self.gelu1 = nn.GELU()
        # 1x1 Conv to implement a linear transform (per time frame)
        self.linear1 = nn.Conv1d(d_model, d_model, kernel_size=1)

        # 2) GLU branch: merge the original input and the S4D branch output
        # Concatenate along the channel dimension and use a 1x1 conv to produce 2*d_model channels,
        # then apply GLU (which splits channels into two halves, one as gate)
        self.glu_conv = nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)  # Apply GLU along the channel dimension

        # 3) Second feed-forward block: LN -> Linear -> GELU -> Linear
        self.ln_ff2 = nn.LayerNorm(d_model)
        self.ff2_linear1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.ff2_gelu = nn.GELU()
        self.ff2_linear2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x: Tensor of shape (batch, channels, time), where channels == d_model.
        Returns:
            B: Processed tensor with the same shape as x.
        """
        # --- Step 1: Apply LN -> S4D -> GELU -> Linear ---
        # Transpose to (B, time, channels) for LayerNorm
        y = x.transpose(1, 2)
        y = self.ln_s4d(y)  # Apply LayerNorm along channel dimension
        y = y.transpose(1, 2)  # Transpose back to (B, channels, time)

        # Pass through S4D layer (returns output and dummy state)
        y, _ = self.s4d(y)

        # Apply GELU and then a 1x1 conv (i.e., linear transformation)
        y = self.gelu1(y)
        y = self.linear1(y)

        # --- Step 2: Merge via GLU ---
        # Concatenate original input and processed output along channel dimension
        cat_xy = torch.cat([x, y], dim=1)  # (B, 2*d_model, time)
        z = self.glu_conv(cat_xy)  # (B, 2*d_model, time)
        z = self.glu(z)  # GLU reduces channel dimension back to d_model

        # Skip connection: add the gated output to the original input
        A = x + z

        # --- Step 3: Second feed-forward block ---
        # Apply another LayerNorm and 1x1 convs with GELU in between
        a_ = A.transpose(1, 2)  # (B, time, channels)
        a_ = self.ln_ff2(a_)  # LayerNorm along channels
        a_ = a_.transpose(1, 2)  # Back to (B, channels, time)
        a_ = self.ff2_linear1(a_)  # Linear transform
        a_ = self.ff2_gelu(a_)  # GELU activation
        a_ = self.ff2_linear2(a_)  # Linear transform

        # Final residual addition
        B = A + a_
        return B


# =====================
# Separator
# =====================
class Separator(nn.Module):
    """
    SpeakerBeam-SS Style Separator.

    The overall flow is as follows:
      1) Apply LayerNorm to the encoder output (input: (B, channels, T)).
      2) Project the high-dimensional encoder output (e.g., 4096 channels) to a lower dimension (e.g., 256 channels)
         via a 1x1 convolution.
      3) Process the projected features with a series of (Conv1DBlock + S4DBlock) pairs (first stage).
      4) Apply a multiplicative adaptation layer that fuses speaker embedding (d-vector) information.
      5) Further process with another series of (Conv1DBlock + S4DBlock) pairs (second stage).
      6) Project back to the original encoder channel dimension with a final 1x1 convolution.
      7) Apply final LayerNorm and ReLU.
      8) Combine the final output with the original encoder output via elementwise multiplication.

    This design enables the network to extract target speaker features effectively.
    """

    def __init__(self, channels=4096, num_blocks1=3, num_blocks2=1):
        super().__init__()
        # The internal processing dimension for the separator is 256
        out_channels = 256
        # Intermediate hidden layer channels
        hidden_channels = 512

        # Input LayerNorm (applied on (B, T, C))
        self.layer_norm_in = nn.LayerNorm(channels)

        # 1x1 convolution to project from high-dim encoder output to out_channels (256)
        self.in_conv1x1 = nn.Conv1d(channels, out_channels, kernel_size=1)

        # First stage: Repeated (Conv1DBlock + S4DBlock) blocks
        self.blocks1 = nn.ModuleList()
        for i in range(num_blocks1):
            block = nn.ModuleList([
                Conv1DBlock(
                    in_chan=out_channels,  # 256 channels
                    hid_chan=hidden_channels, # 512 channels
                    skip_out_chan=0,  # no separate skip connection (using residual add instead)
                    kernel_size=3,  # typical kernel size
                    padding=(3 - 1) * (2 ** i),  # padding to maintain causal behavior, scaled by dilation
                    dilation=2 ** i,
                    norm_type="gLN",  # use global LayerNorm within the block
                    causal=True  # causal convolution; lookahead blocks can be made non-causal as needed
                ),
                S4DBlock(d_model=out_channels)  # S4D block with d_model set to 256
            ])
            self.blocks1.append(block)

        # Multiplicative adaptation layer for fusing speaker embedding information.
        # (Note: for multiplicative adaptation, the enrollment embedding dimension should match out_channels.)
        self.adapt = MulAddAdaptLayer(indim=out_channels, enrolldim=out_channels, ninputs=1, do_addition=False)

        # Second stage: further processing with (Conv1DBlock + S4DBlock)
        self.blocks2 = nn.ModuleList()
        for i in range(num_blocks2):
            block = nn.ModuleList([
                Conv1DBlock(
                    in_chan=out_channels,
                    hid_chan=hidden_channels,
                    skip_out_chan=0,
                    kernel_size=3,
                    padding=(3 - 1) * (2 ** i),
                    dilation=2 ** i,
                    norm_type="gLN",
                    causal=True
                ),
                S4DBlock(d_model=out_channels)
            ])
            self.blocks2.append(block)

        # Final 1x1 conv to project back to the original encoder channel dimension (4096)
        self.out_conv1x1 = nn.Conv1d(out_channels, channels, kernel_size=1)

        # Final LayerNorm (applied on (B, T, C))
        self.layer_norm_out = nn.LayerNorm(channels)

    def forward(self, x, spk_embedding):
        """
        Args:
            x: Tensor of shape (batch, channels, time), where channels=4096 (encoder output)
            spk_embedding: Speaker embedding tensor, assumed to have shape (batch, spk_embed_dim)
                           Here, the embedding is expected to be 256-dimensional to match the internal dimension.
        Returns:
            x: Processed latent representation for the decoder, shape (batch, channels, time)
        """
        # Save original encoder output for later residual multiplication
        input_orig = x

        # 1) Apply LayerNorm over channel dimension. Transpose to (B, T, C) for LN.
        x = x.transpose(1, 2)  # (B, T, 4096)
        x = self.layer_norm_in(x)
        x = x.transpose(1, 2)  # (B, 4096, T)

        # 2) Project high-dim encoder output to lower dimension (256 channels)
        x = self.in_conv1x1(x)  # (B, 256, T)

        # 3) Process through first stage blocks (Conv1DBlock + S4DBlock repeated)
        for conv1d_block_1, s4d_block_1 in self.blocks1:
            x = conv1d_block_1(x)  # Process with Conv1DBlock
            x = s4d_block_1(x)  # Process with S4DBlock

        # 4) Apply multiplicative adaptation using speaker embedding.
        #    The embedding is broadcast along the time dimension.
        x = self.adapt(x, spk_embedding)

        # 5) Process through second stage blocks (further refinement)
        for conv1d_block_2, s4d_block_2 in self.blocks2:
            x = conv1d_block_2(x)
            x = s4d_block_2(x)

        # 6) Project back to the original channel dimension (4096) via 1x1 convolution.
        x = self.out_conv1x1(x)

        # 7) Apply final LayerNorm (transpose for LN and transpose back)
        x = x.transpose(1, 2)  # (B, T, 4096)
        x = self.layer_norm_out(x)
        x = x.transpose(1, 2)  # (B, 4096, T)

        # 8) Apply final ReLU non-linearity
        x = F.relu(x)

        # 9) Combine with the original encoder output via elementwise multiplication (residual gating)
        x = x * input_orig

        return x


# =====================
#  Decoder
# =====================
class Decoder(nn.Module):
    """
    Decoder module:

    Converts the separated latent representation back to a time-domain waveform.
    It uses a ConvTranspose1d layer to perform the reconstruction.

    Input shape: (batch, channels, time) where channels is the encoder output dimension (4096)
    Output shape: (batch, 1, reconstructed_time)
    """

    def __init__(self, in_channels=4096, kernel_size=320, stride=160):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        x = self.deconv(x)  # Reconstruct waveform: (B, 1, time')
        return x


# =====================
# SpeakerBeam-SS: Complete Model
# =====================
class SpeakerBeamSS(nn.Module):
    """
    SpeakerBeam-SS model.

    Overall architecture:
      Encoder -> Separator -> Decoder

    - The encoder converts the waveform to a latent representation.
    - The separator refines the latent representation using target speaker information
        (via speaker embeddings, multiplicative adaptation, and S4D blocks).
    - The decoder converts the refined latent representation back to a time-domain waveform.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()  # Maps waveform to latent space (4096 channels)
        self.separator = Separator()  # Processes latent representation (projects to 256, processes, then projects back)
        self.decoder = Decoder()  # Reconstructs waveform from latent representation

    def forward(self, mixture, enrollment):
        """
        Args:
            mixture: Tensor of shape (batch, 1, time) -- the input mixed waveform.
            enrollment: Tensor of shape (batch, 1, time') -- target speaker's enrollment waveform.
                        (The speaker embedding extraction is assumed to be handled inside adapt layer or externally.)
        Returns:
            out_wav: Tensor of shape (batch, 1, reconstructed_time) -- the separated target speech waveform.
        """
        # Encode the input mixture into latent representation.
        enc_out = self.encoder(mixture)  # (B, 4096, time')
        # Process latent representation with the separator.
        sep_out = self.separator(enc_out, enrollment)  # (B, 4096, time')
        # Decode the latent representation back to a waveform.
        out_wav = self.decoder(sep_out)  # (B, 1, reconstructed_time)
        return out_wav


# =====================
# example
# =====================
if __name__ == "__main__":
    import torchaudio

    # 1つ目の音声をロード
    waveform1, sample_rate1 = torchaudio.load("../data/20250306170609.wav")
    if sample_rate1 != 16000:
        waveform1 = torchaudio.transforms.Resample(orig_freq=sample_rate1, new_freq=16000)(waveform1)

    # 2つ目の音声をロード
    waveform2, sample_rate2 = torchaudio.load("../data/20250306170609.wav")  # 別のファイル
    if sample_rate2 != 16000:
        waveform2 = torchaudio.transforms.Resample(orig_freq=sample_rate2, new_freq=16000)(waveform2)

    # バッチ化（異なる長さの場合は padding が必要）
    max_length = max(waveform1.shape[1], waveform2.shape[1])
    waveform1 = torch.nn.functional.pad(waveform1, (0, max_length - waveform1.shape[1]))
    waveform2 = torch.nn.functional.pad(waveform2, (0, max_length - waveform2.shape[1]))

    # バッチサイズ2の Tensor にする
    batch_waveform = torch.stack([waveform1, waveform2], dim=0)
    speaker_embeddings = get_speaker_embeddings_batch(batch_waveform)

    batch_size = 2
    input_len = 16000  # 1秒分 @16kHz 32bit float
    mixture = torch.randn(batch_size, 1, input_len)

    model = SpeakerBeamSS()
    with torch.no_grad():
        out = model(mixture, speaker_embeddings)
    print("Input:", mixture.shape, "Output:", out.shape)
