"""
Lección: 08-llava-onevision-single-multi-video
Fase: 12
LLaVA-OneVision (Li 2024): single model unified image, multi-image, video.
Sub-image tiling + frame sampling. OneVision projection.
"""
from __future__ import annotations
import numpy as np


def sample_video_frames(video, n_frames=8, strategy="uniform"):
    """video: (T, H, W, C) -> (n_frames, H, W, C)."""
    T = video.shape[0]
    if strategy == "uniform":
        idx = np.linspace(0, T - 1, n_frames).astype(int)
    elif strategy == "random":
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(T, n_frames, replace=False))
    else:
        idx = np.arange(n_frames)
    return video[idx]


def onevision_tile_image(image, max_tiles=4):
    """image (H, W, C) -> tiles (n_tiles, 336, 336, C)"""
    # use aspect ratio logic simplified
    H, W, C = image.shape
    if H == W:
        n_h, n_w = 1, 1
    elif H > W:
        n_h = max_tiles
        n_w = 1
    else:
        n_h = 1
        n_w = max_tiles
    # crop
    cropped = image[: (H // n_h) * n_h, : (W // n_w) * n_w, :]
    th, tw = H // n_h, W // n_w
    return cropped.reshape(n_h, th, n_w, tw, C).transpose(0, 2, 1, 3, 4).reshape(n_h * n_w, th, tw, C)


def onevision_tile_video(video, n_frames=8, max_tiles_per_frame=4):
    """Video -> tiled frames. (n_frames * n_tiles, 336, 336, C)."""
    frames = sample_video_frames(video, n_frames=n_frames, strategy="uniform")
    all_tiles = []
    for f in frames:
        tiles = onevision_tile_image(f, max_tiles=max_tiles_per_frame)
        all_tiles.append(tiles)
    return np.concatenate(all_tiles, axis=0)


def encode_tiles(tiles, embed_dim=1024):
    """Simula ViT encode: (n_tiles, h, w, C) -> (n_tiles, n_patches + 1, embed_dim)."""
    rng = np.random.default_rng(hash(tiles.shape) & 0xFFFFFFFF)
    n_tiles = tiles.shape[0]
    H = tiles.shape[1]
    patch_size = 14
    n = (H // patch_size) ** 2 + 1
    return rng.standard_normal((n_tiles, n, embed_dim)) * 0.1


def onevision_forward(image=None, images=None, video=None,
                      llm_dim=4096, embed_dim=1024):
    """Single unified forward. Returns total tokens."""
    tokens_list = []
    if image is not None:
        tiles = onevision_tile_image(image, max_tiles=4)
        feats = encode_tiles(tiles, embed_dim=embed_dim)
        tokens_list.append(feats.reshape(-1, embed_dim))
    if images is not None:
        for img in images:
            tiles = onevision_tile_image(img, max_tiles=1)
            feats = encode_tiles(tiles, embed_dim=embed_dim)
            tokens_list.append(feats.reshape(-1, embed_dim))
    if video is not None:
        tiles = onevision_tile_video(video, n_frames=8, max_tiles_per_frame=1)
        feats = encode_tiles(tiles, embed_dim=embed_dim)
        tokens_list.append(feats.reshape(-1, embed_dim))
    if not tokens_list:
        return None
    return np.concatenate(tokens_list, axis=0)


def main() -> int:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((672, 1008, 3))
    tiles = onevision_tile_image(img, max_tiles=2)
    print(f"Image tiles: {tiles.shape}")
    video = rng.standard_normal((30, 224, 224, 3))
    frames = sample_video_frames(video, n_frames=8)
    print(f"Video frames: {frames.shape}")
    out = onevision_forward(image=img, video=video, embed_dim=1024, llm_dim=4096)
    print(f"Total tokens: {out.shape}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())