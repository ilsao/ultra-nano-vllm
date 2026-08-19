import unittest

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class SamplingParamsTests(unittest.TestCase):
    def test_accepts_zero_temperature(self):
        self.assertEqual(SamplingParams(temperature=0).temperature, 0)

    def test_rejects_invalid_nonzero_temperatures(self):
        for temperature in (-1.0, 1e-12):
            with self.subTest(temperature=temperature), self.assertRaises(
                ValueError
            ):
                SamplingParams(temperature=temperature)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.forward = Sampler.forward.__wrapped__

    def test_zero_temperature_selects_argmax(self):
        logits = torch.tensor([[1.0, 4.0, 2.0], [3.0, -1.0, 2.0]])
        temperatures = torch.zeros(2)

        tokens = self.forward(Sampler(), logits, temperatures)

        torch.testing.assert_close(tokens, torch.tensor([1, 0]))

    def test_mixed_batch_preserves_positive_temperature_sampling(self):
        logits = torch.tensor([[1.0, 4.0, 2.0], [0.5, 1.5, -0.5]])
        temperatures = torch.tensor([0.0, 0.6])

        torch.manual_seed(7)
        safe_temperatures = torch.tensor([1.0, 0.6])
        probs = torch.softmax(
            logits / safe_temperatures.unsqueeze(dim=1),
            dim=-1,
        )
        sampled = probs.div(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        expected = torch.where(
            temperatures == 0,
            logits.argmax(dim=-1),
            sampled,
        )

        torch.manual_seed(7)
        tokens = self.forward(Sampler(), logits, temperatures)

        torch.testing.assert_close(tokens, expected)


if __name__ == "__main__":
    unittest.main()
