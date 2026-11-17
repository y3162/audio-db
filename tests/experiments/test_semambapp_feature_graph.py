"""The discriminator family shares the view at the waveform junction."""

import unittest

import torch
from torch import nn

from src.models.semambapp.discriminator import (
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)


class RecordingDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, waveform):
        self.inputs.append(waveform)
        return waveform, [waveform]


class TestFeatureGraph(unittest.TestCase):
    def test_family_shares_real_and_fake_channel_views(self):
        for family in (MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator):
            with self.subTest(family=family.__name__):
                module = family.__new__(family)
                nn.Module.__init__(module)
                module.discriminators = nn.ModuleList([RecordingDiscriminator() for _ in range(3)])
                real = torch.randn(1, 32)
                fake = torch.randn(1, 32, requires_grad=True)
                module(real, fake)
                for index in (0, 1):
                    first = module.discriminators[0].inputs[index]
                    self.assertEqual(first.shape, (1, 1, 32))
                    for child in module.discriminators[1:]:
                        self.assertIs(first, child.inputs[index])
                self.assertIsNot(module.discriminators[0].inputs[0], module.discriminators[0].inputs[1])


if __name__ == '__main__':
    unittest.main()
