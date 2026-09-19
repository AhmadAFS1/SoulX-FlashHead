"""Opt-in delivery-only mouth super-resolution, independent of SoulX recurrence.

Share one SRVGGUpscaler across sessions on the engine's owner thread, but create
one MouthEnhancer per session. Feed RGB uint8 frames from Engine.generate and
close the enhancer when the call ends. No face warp or pixel-history averaging.
"""
import math
import cv2
import numpy as np
import torch
from .vendor.srvgg import SRVGGNetCompact


class SRVGGUpscaler:
    """Official general-x4v3 weights, optionally resized to 1x/2x delivery.

    This is a native 4x model, not Ojin's unspecified 2x checkpoint/engine or
    proprietary mouth refiner.
    No downloads or dependency installation happen during construction.
    """
    def __init__(self, weights, device='cuda'):
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.model = SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint.get('params_ema', checkpoint.get('params', checkpoint)), strict=True)
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def upscale(self, rgb, scale=2):
        if scale not in (1, 2, 4):
            raise ValueError('Output scale must be 1, 2 or 4')
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
        tensor = tensor.to(device=self.device, dtype=self.dtype) / 255.
        output = self.model(tensor).float().clamp_(0, 1)
        output = output[0].permute(1, 2, 0).cpu().numpy()
        if scale != 4:
            output = cv2.resize(output, (rgb.shape[1]*scale, rgb.shape[0]*scale),
                                interpolation=cv2.INTER_AREA)
        return (output*255).round().clip(0, 255).astype(np.uint8)


def feather_mask(points, box):
    """Padded, feathered current lip hull in crop-local coordinates."""
    x0, y0, x1, y1 = box
    local = np.asarray(points, dtype=np.float32) - (x0, y0)
    width = float(np.ptp(local[:, 0]))
    mask = np.zeros((y1-y0, x1-x0), np.float32)
    hull = cv2.convexHull(np.round(local).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.)
    radius = max(1, round(width*.13))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1., width*.08))
    # Explicit zero at crop boundary prevents a visible rectangular seam.
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    edge = np.minimum.reduce([xx, yy, mask.shape[1]-1-xx, mask.shape[0]-1-yy])
    mask *= np.clip(edge/max(2., width*.12), 0, 1)
    return np.clip(mask, 0, 1)


class MouthEnhancer:
    """Causal ROI tracking and soft composition; own one instance per stream."""
    OUTER_LIP = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146)

    def __init__(self, upscaler, *, scale=1, strength=.65, smoothing=.65, detector=None):
        if scale not in (1, 2):
            raise ValueError('Mouth output scale must be 1 or 2')
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Strength must be finite and in [0, 1]')
        if not math.isfinite(smoothing) or not 0 < smoothing <= 1:
            raise ValueError('Smoothing must be finite and in (0, 1]')
        self.upscaler, self.scale = upscaler, scale
        self.strength, self.smoothing = strength, smoothing
        self.previous = None
        self.mesh = None
        if detector is None and strength:
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                max_num_faces=1, refine_landmarks=True, min_detection_confidence=.5,
                min_tracking_confidence=.5)
            detector = self._detect
        self.detector = detector

    def _detect(self, rgb):
        prediction = self.mesh.process(rgb)
        if not prediction.multi_face_landmarks:
            return None
        marks = prediction.multi_face_landmarks[0].landmark
        return np.array([(marks[i].x*rgb.shape[1], marks[i].y*rgb.shape[0])
                         for i in self.OUTER_LIP], dtype=np.float32)

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None
        self.previous = None

    def process_frame(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected HWC RGB uint8')
        h, w = rgb.shape[:2]
        scale = self.scale
        result = (rgb.copy() if scale == 1 else
                  cv2.resize(rgb, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC))
        if not self.strength:
            return result, {'detected': False, 'applied': False}
        points = self.detector(rgb)
        if (points is None or np.shape(points) != (20, 2) or not np.isfinite(points).all()
                or np.ptp(points[:, 0]) < 8):
            self.previous = None
            return result, {'detected': False, 'applied': False}
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo+hi)/2
        width = hi[0]-lo[0]
        desired = np.array([*center, max(48., width*1.15),
                            max(32., width*.65, (hi[1]-lo[1])*1.5)])
        if self.previous is not None and np.linalg.norm(center-self.previous[:2]) < width*.75:
            region = self.smoothing*desired + (1-self.smoothing)*self.previous
        else:
            region = desired
        self.previous = region
        cx, cy, rx, ry = region
        box = (max(0, int(np.floor(cx-rx))), max(0, int(np.floor(cy-ry))),
               min(w, int(np.ceil(cx+rx))), min(h, int(np.ceil(cy+ry))))
        x0, y0, x1, y1 = box
        if x1-x0 < 8 or y1-y0 < 8 or lo[0] < x0 or lo[1] < y0 or hi[0] >= x1 or hi[1] >= y1:
            self.previous = None
            return result, {'detected': True, 'applied': False}
        mask = feather_mask(points, box)
        restored = self.upscaler.upscale(rgb[y0:y1, x0:x1], scale)
        if restored.shape != ((y1-y0)*scale, (x1-x0)*scale, 3):
            raise ValueError('Unexpected upscaler output geometry')
        if scale != 1:
            mask = cv2.resize(mask, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = mask[:, :, None]*self.strength
        original = result[y0*scale:y1*scale, x0*scale:x1*scale]
        composed = original.astype(np.float32)*(1-alpha) + restored.astype(np.float32)*alpha
        original[:] = composed.round().clip(0, 255).astype(np.uint8)
        return result, {'detected': True, 'applied': True, 'box': box}

    def process_chunk(self, frames):
        return np.stack([self.process_frame(frame)[0] for frame in frames])
