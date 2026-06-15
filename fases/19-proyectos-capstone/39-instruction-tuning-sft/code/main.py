"""
Lección: 39-instruction-tuning-sft
Fase: 19
Capstone de ingeniería AI: 39 Instruction Tuning Sft.
"""
from __future__ import annotations
import sys

def sft_train(model_id, dataset_id, output_dir):
    """SFT trainer."""
    from trl import SFTTrainer, SFTConfig
    model, tokenizer = load_pretrained(model_id)
    config = SFTConfig(output_dir=output_dir, num_train_epochs=3,
                      per_device_train_batch_size=2,
                      gradient_accumulation_steps=8)
    trainer = SFTTrainer(model, config, tokenizer, dataset_id)
    trainer.train()
    return model



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
