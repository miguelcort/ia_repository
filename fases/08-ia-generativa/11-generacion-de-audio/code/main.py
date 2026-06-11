"""
Lección: 11-generacion-de-audio
Fase: 08
Audio generation: VALL-E, MusicGen, AudioLDM, Bark, Stable Audio.
"""
from __future__ import annotations
import sys
import numpy as np


def mel_spectrogram_mock(audio, n_mels=80, n_fft=1024, hop=256, sample_rate=22050):
    """Mock mel spectrogram. Real: STFT -> mel filterbank -> log.
    audio: 1D array. Returns: (T, n_mels).
    """
    n = len(audio)
    n_frames = n // hop
    # Mock: random projection
    rng = np.random.default_rng(0)
    mel_basis = rng.standard_normal((n_frames, n_mels)) * 0.1
    return np.log(np.abs(mel_basis) + 1e-6)


def compress_mel(mel, vae_encode_W):
    """Compress mel a latente (VAE encoder)."""
    return mel @ vae_encode_W


def decoder_diffusion(latent, alpha_bar, t, eps_pred):
    """Diffusion decoder: eps_pred -> denoised latent -> mel."""
    a = alpha_bar[t]
    return latent * np.sqrt(a) + (1 - a) * eps_pred  # mock


def text_encoder_mock(text_tokens, W_text):
    """Text encoder: tokens -> text embedding."""
    return text_tokens @ W_text


def vall_e_components():
    """VALL-E: neural codec LM para TTS."""
    return {
        "Codec": "Encodec: audio 24kHz -> 8 codebooks, 75 tokens/sec",
        "LM": "Decoder transformer, predict next codec tokens",
        "Prompt": "3-10s reference audio, voice cloning",
        "Text": "Phoneme sequence, prompt para TTS",
        "Output": "Codec tokens -> audio via codec decoder",
    }


def musicgen_components():
    """MusicGen: text-to-music."""
    return {
        "Encoder": "EnCodec audio 32kHz, 4 codebooks",
        "Text": "T5 text encoder",
        "LM": "Transformer decoder con cross-attn",
        "Inference": "50 steps sampling, classifier-free guidance",
        "Output": "30s music, 32kHz",
    }


def audio_models():
    """Audio generation models SOTA."""
    return {
        "VALL-E / VALL-E X": "TTS con voice cloning, Microsoft",
        "MusicGen": "Text-to-music, Meta 2023",
        "AudioLDM / AudioLDM 2": "Text-to-audio latent diffusion",
        "Bark": "Multilingual TTS, suno-ai",
        "Stable Audio": "Text-to-audio, Stability AI",
        "Suno / Udio": "End-to-end song generation",
    }


def main() -> int:
    print("=== Audio generation models ===")
    for k, v in audio_models().items():
        print(f"  {k:25s} {v}")
    # Demo
    audio = np.random.default_rng(0).standard_normal(22050)  # 1s
    mel = mel_spectrogram_mock(audio)
    print(f"\nAudio 1s: mel {mel.shape} (T, 80 mels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())