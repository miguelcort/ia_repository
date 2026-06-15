"""
Lección: 12-video-understanding-pipeline
Fase: 19
Capstone de ingeniería AI: 12 Video Understanding Pipeline.
"""
from __future__ import annotations
import sys

def extract_frames(video_path, fps=1):
    """Extract frames at fps."""
    import av
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        if frame.time >= len(frames) / fps:
            frames.append(frame.to_image())
    return frames


def video_qa(video_path, question, vlm):
    """VLM QA."""
    return vlm.chat(video=video_path, question=question)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
