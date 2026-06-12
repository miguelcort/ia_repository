"""
Lección: 09-production-quantization
Fase: 17
Production quantization: INT8, INT4, FP8,
FP4, GPTQ, AWQ, GGUF, mixed precision,
calibration, accuracy preservation.
"""
from __future__ import annotations


QUANT_FORMATS = {
    "FP16": {"bits": 16, "method": "native", "compression": 1.0, "accuracy_loss": 0.0},
    "BF16": {"bits": 16, "method": "native", "compression": 1.0, "accuracy_loss": 0.0},
    "INT8": {"bits": 8, "method": "PTQ", "compression": 2.0, "accuracy_loss": 0.005},
    "FP8": {"bits": 8, "method": "PTQ", "compression": 2.0, "accuracy_loss": 0.002},
    "INT4": {"bits": 4, "method": "GPTQ/AWQ", "compression": 4.0, "accuracy_loss": 0.02},
    "FP4": {"bits": 4, "method": "NVFP4", "compression": 4.0, "accuracy_loss": 0.03},
    "INT2": {"bits": 2, "method": "aggressive", "compression": 8.0, "accuracy_loss": 0.10},
    "GGUF_Q4": {"bits": 4, "method": "GGUF", "compression": 4.0, "accuracy_loss": 0.025},
    "GGUF_Q5": {"bits": 5, "method": "GGUF", "compression": 3.2, "accuracy_loss": 0.015},
    "GGUF_Q8": {"bits": 8, "method": "GGUF", "compression": 2.0, "accuracy_loss": 0.005},
}


def list_formats():
    return list(QUANT_FORMATS.keys())


def get_format(name):
    return QUANT_FORMATS.get(name)


def estimate_size(model_size_b, format_name):
    """Estimate size in GB after quantization."""
    fmt = QUANT_FORMATS.get(format_name)
    if not fmt:
        return None
    bytes_per_param = fmt["bits"] / 8
    return model_size_b * bytes_per_param


def pick_format(target_compression, max_loss=0.05):
    """Pick a format meeting compression target with acceptable loss."""
    candidates = []
    for name, info in QUANT_FORMATS.items():
        if info["compression"] >= target_compression and info["accuracy_loss"] <= max_loss:
            candidates.append((name, info))
    if not candidates:
        return None
    return min(candidates, key=lambda kv: kv[1]["accuracy_loss"])


def main() -> int:
    print(f"Formats: {len(QUANT_FORMATS)}")
    print(f"70B FP16: {estimate_size(70, 'FP16')}GB")
    print(f"70B INT4: {estimate_size(70, 'INT4')}GB")
    print(f"Pick 4x: {pick_format(4.0)[0]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())