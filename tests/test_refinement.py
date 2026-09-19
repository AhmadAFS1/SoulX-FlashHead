"""CPU-only sampler invariants; no GPU quality claims."""
import unittest
import torch
from soulx_rtc.refinement import refine_clean_latent


class RefinementTests(unittest.TestCase):
    def test_flow_oracle_prefix_and_rng(self):
        clean = torch.randn(2, 3, 5, 2, 2)
        original = clean.clone()
        motions = [torch.ones(3, 1, 2, 2), torch.ones(3, 2, 2, 2) * 2]
        target = clean * 0.5
        base = torch.Generator().manual_seed(50)
        before = base.get_state().clone()
        calls = []
        def oracle(noisy, time):
            for i, motion in enumerate(motions):
                torch.testing.assert_close(noisy[i, :, :motion.shape[1]], motion)
            calls.append(float(time))
            return (noisy - target) / (time / 1000)
        def run():
            return refine_clean_latent(clean, [torch.tensor([350.]), torch.tensor([200.])],
                1000, motions, [torch.Generator().manual_seed(i) for i in (1, 2)], oracle)
        result = run()
        torch.testing.assert_close(result, run())
        self.assertEqual(calls, [350., 200., 350., 200.])
        for i, motion in enumerate(motions):
            torch.testing.assert_close(result[i, :, :motion.shape[1]], motion)
            torch.testing.assert_close(result[i, :, motion.shape[1]:], target[i, :, motion.shape[1]:])
        self.assertTrue(torch.equal(clean, original))
        self.assertTrue(torch.equal(base.get_state(), before))

    def test_disabled_and_invalid(self):
        clean = torch.zeros(1, 1, 2, 1, 1)
        self.assertIs(refine_clean_latent(clean, [], 1000, [], [], None), clean)
        for times in ([0.], [1000.], [100., 200.]):
            with self.assertRaises(ValueError):
                refine_clean_latent(clean, [torch.tensor([t]) for t in times], 1000,
                    [clean[0, :, :1]], [torch.Generator()], None)


if __name__ == '__main__':
    unittest.main()
