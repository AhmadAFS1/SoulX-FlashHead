"""CPU-only geometry/fallback contracts; no model-quality claims."""
import unittest
import numpy as np
from soulx_rtc.mouth_sr import MouthEnhancer


class WhiteUpscaler:
    def upscale(self, rgb, scale):
        return np.full((rgb.shape[0]*scale, rgb.shape[1]*scale, 3), 255, np.uint8)


def lip_points(cx=64, cy=60):
    angle = np.linspace(0, 2*np.pi, 20, endpoint=False)
    return np.stack([cx+20*np.cos(angle), cy+5*np.sin(angle)], axis=1)


class MouthSRTests(unittest.TestCase):
    def test_input_background_and_mask(self):
        frame = np.full((128, 128, 3), 50, np.uint8)
        before = frame.copy()
        enhancer = MouthEnhancer(WhiteUpscaler(), detector=lambda _: lip_points())
        output, meta = enhancer.process_frame(frame)
        self.assertTrue(meta['applied'])
        self.assertTrue(np.array_equal(frame, before))
        x0,y0,x1,y1 = meta['box']; mask = np.ones(frame.shape[:2], bool)
        mask[y0:y1,x0:x1] = False
        self.assertTrue(np.array_equal(frame[mask], output[mask]))
        self.assertGreater(int(output[60,64,0]), 50)
        self.assertLess(int(output[60,64,0]), 255)
        self.assertTrue(np.array_equal(output[y0, x0:x1], frame[y0, x0:x1]))

    def test_no_face_no_effect_and_reset(self):
        frame = np.zeros((128,128,3), np.uint8)
        enhancer = MouthEnhancer(WhiteUpscaler(), detector=lambda _: None)
        enhancer.previous = np.ones(4)
        output, meta = enhancer.process_frame(frame)
        self.assertFalse(meta['applied']); self.assertIsNone(enhancer.previous)
        self.assertTrue(np.array_equal(output, frame))
        disabled = MouthEnhancer(WhiteUpscaler(), strength=0)
        self.assertTrue(np.array_equal(disabled.process_frame(frame)[0], frame))

    def test_scale_and_chunk_partition(self):
        frames = np.full((5,128,128,3), 80, np.uint8)
        def detector():
            trajectory = iter([64, 67, 70, 74, 78])
            return lambda _: lip_points(next(trajectory))
        a = MouthEnhancer(WhiteUpscaler(), scale=2, detector=detector())
        b = MouthEnhancer(WhiteUpscaler(), scale=2, detector=detector())
        together = a.process_chunk(frames)
        split = np.concatenate([b.process_chunk(frames[:2]), b.process_chunk(frames[2:])])
        self.assertEqual(together.shape, (5,256,256,3))
        self.assertTrue(np.array_equal(together, split))

    def test_border_and_invalid_detection(self):
        frame = np.zeros((128,128,3), np.uint8)
        for points in (lip_points(2,2), np.full((20,2), np.nan), np.ones((4,2))):
            enhancer = MouthEnhancer(WhiteUpscaler(), detector=lambda _, p=points:p)
            output, meta = enhancer.process_frame(frame)
            self.assertEqual(output.shape, frame.shape)
            self.assertFalse(meta['applied'])
        with self.assertRaises(ValueError):
            MouthEnhancer(None, strength=float('nan'))


if __name__ == '__main__':
    unittest.main()
