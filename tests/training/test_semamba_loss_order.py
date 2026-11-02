import unittest

import torch

from tests.training.four_process.real_models import _semamba_discriminator


class _Scores:
    def __init__(self, real_scores, fake_scores):
        self.real_scores = real_scores
        self.fake_scores = fake_scores

    def __call__(self, waveform, paired=None):
        if paired is not None:
            return self.real_scores, self.fake_scores, [], []
        return (self.real_scores if waveform.item() == 1 else self.fake_scores), []


class SEMambaLossOrderTest(unittest.TestCase):
    def test_discriminator_families_interleave_real_and_fake(self):
        from src.training.se.semamba_disc import MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator

        for cls in (MultiScaleSubbandCQTDiscriminator, MultiResolutionDiscriminator):
            calls = []

            class Recording(torch.nn.Module):
                def __init__(self, index):
                    super().__init__()
                    self.index = index

                def forward(self, waveform):
                    calls.append((self.index, waveform.item()))
                    return waveform, [waveform]

            model = cls.__new__(cls)
            torch.nn.Module.__init__(model)
            model.discriminators = torch.nn.ModuleList([Recording(0), Recording(1)])
            real, fake, real_features, fake_features = model(torch.tensor(1.0), torch.tensor(2.0))
            self.assertEqual(calls, [(0, 1), (0, 2), (1, 1), (1, 2)])
            self.assertEqual([x.item() for x in real], [1, 1])
            self.assertEqual([x.item() for x in fake], [2, 2])
            self.assertEqual(len(real_features), len(fake_features))

    def test_discriminator_sums_each_family_before_combining(self):
        # Each small MRD term rounds away if accumulated after the large CQT
        # term; the released solver sums the two discriminator families first.
        band = _Scores([torch.tensor([-9999.0])], [torch.tensor([0.0])])
        mrd = _Scores([torch.tensor([0.0])] * 6, [torch.tensor([0.0])] * 6)
        loss = _semamba_discriminator("independent", band, mrd, torch.tensor(1.0), torch.tensor(0.0))
        self.assertEqual(loss.item(), 100000008.0)


if __name__ == "__main__":
    unittest.main()
