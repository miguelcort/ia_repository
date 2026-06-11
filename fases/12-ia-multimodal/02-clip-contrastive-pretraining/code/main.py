"""
Lección: 02-clip-contrastive-pretraining
Fase: 12
CLIP (OpenAI 2021): image encoder + text encoder, contrastive learning.
InfoNCE sobre N x N pairs. Similitud coseno sobre temperature.
"""
from __future__ import annotations
import sys
import numpy as np


def l2_normalize(x, eps=1e-12):
    """L2 normalize rows of x. (n, d) -> (n, d)."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_sim_matrix(a, b):
    """a: (n, d), b: (m, d) -> (n, m)."""
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return a_n @ b_n.T


def clip_contrastive_loss(image_emb, text_emb, temperature=0.07):
    """InfoNCE: diagonal = positive pairs. (n, d) -> scalar loss.
    Returns (loss, logit_image_to_text) where logit has shape (n, n).
    """
    logits = cosine_sim_matrix(image_emb, text_emb) / temperature
    n = logits.shape[0]
    labels = np.arange(n)
    # cross-entropy row-wise y column-wise
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    log_softmax_image = np.log(e / e.sum(axis=-1, keepdims=True))
    loss_image = -log_softmax_image[np.arange(n), labels].mean()
    log_softmax_text = np.log(e / e.sum(axis=0, keepdims=True))
    loss_text = -log_softmax_text[np.arange(n), labels].mean()
    return 0.5 * (loss_image + loss_text), logits


def clip_accuracy(logits, k=1):
    """Top-k retrieval accuracy. logits: (n, n) where row i is image i, column j is text j.
    Returns (img2text@k, text2img@k)."""
    n = logits.shape[0]
    # img2text: rank of column i in row i
    correct_img = 0
    for i in range(n):
        # rank of positive
        rank = (logits[i] > logits[i, i]).sum()
        if rank < k:
            correct_img += 1
    # text2img: rank of row i in column i
    correct_text = 0
    for i in range(n):
        rank = (logits[:, i] > logits[i, i]).sum()
        if rank < k:
            correct_text += 1
    return correct_img / n, correct_text / n


def zero_shot_classify(image_emb, class_emb, class_names):
    """image_emb: (d,), class_emb: (n_classes, d). Returns class name."""
    sims = l2_normalize(image_emb.reshape(1, -1)) @ l2_normalize(class_emb).T
    idx = int(sims.argmax())
    return class_names[idx]


def main() -> int:
    rng = np.random.default_rng(0)
    n = 4
    d = 16
    img = rng.standard_normal((n, d))
    txt = rng.standard_normal((n, d))
    loss, logits = clip_contrastive_loss(img, txt)
    print(f"CLIP contrastive loss: {loss:.4f}")
    print(f"Logits shape: {logits.shape}")
    img2t, t2i = clip_accuracy(logits)
    print(f"Top-1: img2txt={img2t:.2f}, txt2img={t2i:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())