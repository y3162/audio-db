import unittest

import torch

from src.se.models.stupase.flow.compat import RotaryEmbedding, apply_rotary_pos_emb


class RotaryTests(unittest.TestCase):
    def test_frequencies_match_x_transformers_2_14_2_layout(self):
        embedding = RotaryEmbedding(8)
        freqs, scale = embedding.forward_from_seq_len(4)
        self.assertEqual(tuple(freqs.shape), (1, 4, 8))
        self.assertEqual(scale, 1.0)
        # Adjacent pairs share an angle. A half-split table would not.
        self.assertTrue(torch.equal(freqs[..., 0::2], freqs[..., 1::2]))
        self.assertFalse(torch.equal(freqs[..., :4], freqs[..., 4:]))

    def test_rotation_swaps_adjacent_pairs(self):
        embedding = RotaryEmbedding(4)
        freqs, scale = embedding.forward_from_seq_len(2)
        values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]])
        rotated = apply_rotary_pos_emb(values, freqs, scale)
        paired = values.reshape(1, 1, 2, 2, 2)
        first, second = paired.unbind(-1)
        adjacent = torch.stack((-second, first), dim=-1).reshape_as(values)
        table = freqs.unsqueeze(1)
        expected = values * table.cos() + adjacent * table.sin()
        self.assertTrue(torch.allclose(rotated, expected))
        # Position 0 is the identity. Position 1 is not a half-vector swap.
        self.assertTrue(torch.allclose(rotated[:, :, 0], values[:, :, 0]))
        half = torch.cat((-values[:, :, 1, 2:], values[:, :, 1, :2]), dim=-1)
        self.assertFalse(torch.allclose(rotated[:, :, 1], half))


if __name__ == '__main__':
    unittest.main()
