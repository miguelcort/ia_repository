"""
Lección: 09-qwen-vl-family-dynamic-fps
Fase: 12
Qwen-VL (Alibaba 2023-2024): dynamic resolution + dynamic FPS video.
Qwen-VL, Qwen2-VL, Qwen2.5-VL. 2D-RoPE position encoding.
"""
from __future__ import annotations
import numpy as np


def dynamic_resolution_tiles(image, max_tiles=4, min_tiles=1):
    """Image -> tiles preservando aspect ratio. Hasta max_tiles.
    Returns list of tile crops (H_tile, W_tile, C)."""
    H, W, C = image.shape
    aspect = W / H
    best = (1, 1)
    best_score = float("inf")
    for n in range(min_tiles, max_tiles + 1):
        for nh in range(1, n + 1):
            nw = n // nh
            if nh * nw != n:
                continue
            score = abs(np.log((nw / nh) / aspect))
            if score < best_score:
                best = (nh, nw)
                best_score = score
    nh, nw = best
    th, tw = H // nh, W // nw
    cropped = image[: th * nh, : tw * nw, :]
    return cropped.reshape(nh, th, nw, tw, C).transpose(0, 2, 1, 3, 4).reshape(nh * nw, th, tw, C)


def dynamic_fps_sampling(video, target_fps=2.0, max_frames=8, source_fps=24.0):
    """Video -> frames a target_fps. Returns (n, H, W, C)."""
    T, H, W, C = video.shape
    # n = target_fps * duration, capped at max_frames
    duration = T / source_fps
    n = max(1, int(target_fps * duration))
    n = min(max_frames, max(1, n))
    # if T > max_frames and n < max_frames, allow up to T
    n = min(n, T)
    idx = np.linspace(0, T - 1, n).astype(int)
    return video[idx]


def encode_image_tiles(tiles, embed_dim=1280):
    """ViT encode tiles: (n_tiles, h, w, C) -> (n_tiles, n_patches+1, embed_dim)."""
    rng = np.random.default_rng(hash(tiles.shape) & 0xFFFFFFFF)
    n_tiles = tiles.shape[0]
    H = tiles.shape[1]
    patch_size = 14
    n = (H // patch_size) ** 2 + 1
    return rng.standard_normal((n_tiles, n, embed_dim)) * 0.1


def encode_video_frames(frames, embed_dim=1280):
    """Video frames -> tokens."""
    rng = np.random.default_rng(hash(frames.shape) & 0xFFFFFFFF)
    T = frames.shape[0]
    H = frames.shape[1]
    patch_size = 14
    n = (H // patch_size) ** 2 + 1
    return rng.standard_normal((T, n, embed_dim)) * 0.1


def qwen2d_rope_position_emb(height, width, dim, base=10000.0):
    """2D-RoPE position embedding (Qwen2-VL)."""
    # half dim for h, half for w -> total dim
    half = dim // 2
    pos_h = np.arange(height, dtype=np.float32)
    pos_w = np.arange(width, dtype=np.float32)
    inv_freq = 1.0 / (base ** (np.arange(0, half, 2, dtype=np.float32) / half))
    emb_h = np.zeros((height, half))
    for i, p in enumerate(pos_h):
        for j, freq in enumerate(inv_freq):
            emb_h[i, 2 * j] = np.sin(p * freq)
            emb_h[i, 2 * j + 1] = np.cos(p * freq)
    emb_w = np.zeros((width, half))
    for i, p in enumerate(pos_w):
        for j, freq in enumerate(inv_freq):
            emb_w[i, 2 * j] = np.sin(p * freq)
            emb_w[i, 2 * j + 1] = np.cos(p * freq)
    return emb_h, emb_w


def main() -> int:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((672, 1008, 3))
    tiles = dynamic_resolution_tiles(img, max_tiles=4)
    print(f"Qwen2-VL image tiles: {tiles.shape}")
    video = rng.standard_normal((48, 224, 224, 3))
    frames = dynamic_fps_sampling(video, target_fps=2.0, max_frames=8)
    print(f"Qwen2-VL video frames: {frames.shape}")
    emb_h, emb_w = qwen2d_rope_position_emb(24, 24, 128)
    print(f"2D-RoPE: h={emb_h.shape}, w={emb_w.shape}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())