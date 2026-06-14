"""Pruebas para la lección 01 — Instruction following como señal de alineación."""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402


class TestSoftmax(unittest.TestCase):
    def test_suma_uno(self):
        s = main.softmax({"A": 1.0, "B": 2.0, "C": 3.0})
        self.assertAlmostEqual(sum(s.values()), 1.0, places=6)

    def test_no_negativos(self):
        s = main.softmax({"A": -5.0, "B": 0.0, "C": 5.0})
        for v in s.values():
            self.assertGreaterEqual(v, 0.0)

    def test_maximo_domina(self):
        s = main.softmax({"A": 0.0, "B": 0.0, "C": 100.0})
        self.assertGreater(s["C"], 0.99)


class TestLabelerSFT(unittest.TestCase):
    def test_respuesta_en_acciones(self):
        for _ in range(50):
            self.assertIn(main.labeler_sft("p"), main.ACTIONS)

    def test_distribucion_aproximada(self):
        random.seed(0)
        counts: dict[str, int] = {a: 0 for a in main.ACTIONS}
        for _ in range(5000):
            counts[main.labeler_sft("p")] += 1
        # La acción B debería dominar sobre A y C
        self.assertGreater(counts["B"], counts["A"])
        self.assertGreater(counts["B"], counts["C"])


class TestRewardModel(unittest.TestCase):
    def test_rewards_separan_ganador_perdedor(self):
        random.seed(0)
        pairwise = [
            (f"p_{i}", "B", "A") for i in range(50)
        ] + [(f"p_{i}", "B", "C") for i in range(50)]
        r = main.fit_reward_model(pairwise, steps=300)
        self.assertGreater(r["B"], r["A"])
        self.assertGreater(r["B"], r["C"])

    def test_bradley_terry_loss_positiva(self):
        r = {"A": 0.0, "B": 1.0, "C": -1.0}
        self.assertGreater(main.bradley_terry_loss(r), 0.0)


class TestPPOUpdate(unittest.TestCase):
    def test_ppo_con_beta_grande_no_se_aparta_mucho(self):
        random.seed(0)
        sft = {"A": 0.1, "B": 0.7, "C": 0.2}
        sft_logits = {"A": -1.0, "B": 0.5, "C": -1.5}
        rewards = {"A": 0.0, "B": 1.0, "C": 0.0}
        policy = dict(sft_logits)
        for _ in range(20):
            policy = main.ppo_update(policy, sft_logits, rewards, beta=10.0)
        kl = main.kl_divergence(main.softmax(policy), sft)
        self.assertLess(kl, 0.1)

    def test_ppo_sin_beta_explota(self):
        random.seed(0)
        sft = {"A": 0.1, "B": 0.7, "C": 0.2}
        sft_logits = {"A": -1.0, "B": 0.5, "C": -1.5}
        rewards = {"A": 0.0, "B": 0.0, "C": 1.0}  # sesgo a C
        policy = dict(sft_logits)
        for _ in range(50):
            policy = main.ppo_update(policy, sft_logits, rewards, beta=0.0)
        probs = main.softmax(policy)
        # Sin KL, debería concentrarse en C
        self.assertGreater(probs["C"], 0.5)


class TestKLDivergence(unittest.TestCase):
    def test_kl_de_p_a_p_es_cero(self):
        p = {"A": 0.5, "B": 0.5, "C": 0.0}
        self.assertAlmostEqual(main.kl_divergence(p, p), 0.0, places=5)


class TestGeneratePairwiseData(unittest.TestCase):
    def test_estructura_de_pares(self):
        random.seed(0)
        # pedimos más datos para absorber los que se filtran por
        # falta de un "loser" con menor preferencia
        data = main.generate_pairwise_data(100)
        self.assertGreaterEqual(len(data), 20)
        for prompt, w, l in data:
            self.assertIn(w, main.ACTIONS)
            self.assertIn(l, main.ACTIONS)
            self.assertNotEqual(w, l)
            self.assertTrue(prompt.startswith("prompt_"))


class TestMain(unittest.TestCase):
    def test_main_retorna_cero(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        salida = buffer.getvalue()
        self.assertIn("SFT", salida)
        self.assertIn("PPO", salida)


if __name__ == "__main__":
    unittest.main()
