"""
Lección: 17-video-language-temporal-grounding
Fase: 12
Video + language: temporal grounding, dense frame sampling,
moment retrieval, action segmentation. VideoBERT, Video-LLaMA,
VideoChat, InternVideo.
"""
from __future__ import annotations
import numpy as np


def sample_dense_frames(video, fps_sample=4, max_frames=32):
    """Dense frame sampling. video (T, H, W, C) -> (n, H, W, C)."""
    T = video.shape[0]
    n = min(max_frames, max(1, T // fps_sample))
    idx = np.linspace(0, T - 1, n).astype(int)
    return video[idx]


def temporal_grounding(video, query_embedding, n_moments=3, embed_dim=512):
    """Localiza momentos en video que matchean query. Returns (n_moments, 2) start/end."""
    T = video.shape[0]
    # mock: pseudo moment detection
    rng = np.random.default_rng(hash(query_embedding.tobytes()[:64]) & 0xFFFFFFFF)
    moments = []
    for _ in range(n_moments):
        s = rng.integers(0, max(1, T - 10))
        e = s + rng.integers(2, min(10, T - s))
        moments.append((int(s), int(e)))
    return sorted(moments, key=lambda m: m[0])


def moment_retrieval_score(pred_moments, gt_moments, iou_threshold=0.5):
    """R1@iou: 1 si alguna prediccion tiene IoU > threshold con GT."""
    def iou(a, b):
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        inter = max(0, e - s)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0
    for p in pred_moments:
        for g in gt_moments:
            if iou(p, g) >= iou_threshold:
                return 1
    return 0


def encode_video_clip(video, embed_dim=512):
    """Video clip -> (n_frames, embed_dim)."""
    rng = np.random.default_rng(hash(video.tobytes()[:64]) & 0xFFFFFFFF)
    n = video.shape[0]
    return rng.standard_normal((n, embed_dim)) * 0.1


def video_text_similarity(video_emb, text_emb):
    """Cosine sim entre video (n, d) y text (d,). Returns max sim."""
    v = video_emb / np.linalg.norm(video_emb, axis=-1, keepdims=True)
    norm = np.linalg.norm(text_emb)
    if norm < 1e-12:
        return 0.0
    t = text_emb / norm
    sims = v @ t
    return float(sims.max())


def main() -> int:
    rng = np.random.default_rng(0)
    video = rng.standard_normal((100, 32, 32, 3))
    frames = sample_dense_frames(video, fps_sample=4, max_frames=16)
    print(f"Dense frames: {frames.shape}")
    query = rng.standard_normal(512)
    moments = temporal_grounding(video, query, n_moments=3)
    print(f"Moments: {moments}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())