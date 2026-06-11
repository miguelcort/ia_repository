"""
Lección: 06-any-resolution-patch-n-pack
Fase: 12
Any resolution: imagenes divididas en grids, sub-imagenes reescaladas,
concatenadas en sequence, position embeddings aprendidas.
Usado en LLaVA-Next, Qwen-VL dynamic res.
"""
from __future__ import annotations
import sys
import numpy as np


def select_best_grid(image_h, image_w, max_tiles=4, min_tiles=1):
    """Selecciona grid (n_h, n_w) preservando aspect ratio.
    Returns (n_h, n_w) con n_h * n_w <= max_tiles.
    """
    aspect = image_w / image_h
    best = (1, 1)
    best_score = float("inf")
    for n in range(min_tiles, max_tiles + 1):
        for nh in range(1, n + 1):
            nw = n // nh
            if nh * nw != n:
                continue
            grid_aspect = nw / nh
            score = abs(np.log(grid_aspect / aspect))
            if score < best_score:
                best_score = score
                best = (nh, nw)
    return best


def split_into_tiles(image, grid):
    """image (H, W, C) -> tiles (nh*nw, h, w, C)."""
    nh, nw = grid
    H, W, C = image.shape
    th, tw = H // nh, W // nw
    # recortar
    cropped = image[: th * nh, : tw * nw, :]
    return cropped.reshape(nh, th, nw, tw, C).transpose(0, 2, 1, 3, 4).reshape(nh * nw, th, tw, C)


def resize_tile(tile, target_size):
    """Resize tile a target (h, w) via nearest-neighbor (simulado)."""
    th, tw, C = tile.shape
    out_h, out_w = target_size
    # muestreo nearest
    rs = np.linspace(0, th - 1, out_h).astype(int)
    cs = np.linspace(0, tw - 1, out_w).astype(int)
    return tile[rs[:, None], cs[None, :], :]


def pack_tiles_with_thumbnail(tiles, target_size):
    """Primer tile = thumbnail, resto = tiles reescalados."""
    th, tw, C = target_size[0], target_size[1], tiles.shape[-1]
    thumb = resize_tile(tiles[0], target_size)
    rest = np.stack([resize_tile(t, target_size) for t in tiles[1:]])
    return np.stack([thumb] + list(rest))


def vit_encoder(tiles, embed_dim=1024):
    """Simula ViT forward en cada tile -> (n_tiles, n_patches + 1, embed_dim)."""
    rng = np.random.default_rng(hash(tiles.shape) & 0xFFFFFFFF)
    n_tiles = tiles.shape[0]
    H, W = tiles.shape[1], tiles.shape[2]
    patch_size = 14
    n = (H // patch_size) * (W // patch_size) + 1
    return rng.standard_normal((n_tiles, n, embed_dim)) * 0.1


def pack_for_llm(tiles, target_size, embed_dim, llm_dim):
    """Tiles -> ViT -> concat en sequence + thumbnail flag."""
    grid_tiles = pack_tiles_with_thumbnail(tiles, target_size)
    feats = vit_encoder(grid_tiles, embed_dim=embed_dim)
    # flatten: (n_tiles, n_patches + 1, embed_dim) -> (n_tiles * (n_patches+1), embed_dim)
    n_tiles, n_p, ed = feats.shape
    flat = feats.reshape(n_tiles * n_p, ed)
    return flat


def main() -> int:
    img = np.random.default_rng(0).standard_normal((672, 1008, 3))
    grid = select_best_grid(672, 1008, max_tiles=4)
    print(f"Grid seleccionado: {grid} (aspect preservado)")
    tiles = split_into_tiles(img, grid)
    print(f"Tiles: {tiles.shape} (n_tiles, h, w, C)")
    packed = pack_for_llm(tiles, target_size=(336, 336), embed_dim=1024, llm_dim=4096)
    print(f"Packed sequence: {packed.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())