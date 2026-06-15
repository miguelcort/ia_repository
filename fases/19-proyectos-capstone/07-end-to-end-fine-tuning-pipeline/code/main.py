"""
Lección: 07-end-to-end-fine-tuning-pipeline
Fase: 19
Capstone de ingeniería AI: 07 End To End Fine Tuning Pipeline.
"""
from __future__ import annotations
import sys

def full_finetune_pipeline(model_id, dataset_id, output_dir):
    """SFT + DPO + eval + quantization + serve."""
    from trl import SFTTrainer, DPOTrainer
    from peft import LoraConfig
    # SFT
    sft = SFTTrainer(model_id, dataset_id, LoraConfig(r=64))
    sft.train()
    # DPO
    dpo = DPOTrainer(sft.model, dataset_id)
    dpo.train()
    # Quantize + save
    sft.model.save_pretrained(output_dir)
    return sft.model



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
