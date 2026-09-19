"""Opt-in Google Tasks FaceLandmarker mouth tracking; one tracker per stream."""
import math
import numpy as np

from .mouth_sr import MouthEnhancer


class FaceLandmarkerTracker:
    def __init__(self, model_path, fps=25):
        if not math.isfinite(fps) or not 0 < fps <= 1000:
            raise ValueError('FPS must be positive and allow distinct millisecond timestamps')
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.fps = fps
        self.index = 0
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path),
                    delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=.5,
                min_face_presence_confidence=.5,
                min_tracking_confidence=.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False))

    def __call__(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected HWC RGB uint8')
        if self.landmarker is None:
            raise RuntimeError('Tracker is closed')
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        timestamp = round(self.index * 1000 / self.fps)
        self.index += 1
        result = self.landmarker.detect_for_video(image, timestamp)
        if not result.face_landmarks:
            return None
        marks = result.face_landmarks[0]
        return np.array([(marks[i].x * rgb.shape[1], marks[i].y * rgb.shape[0])
                         for i in MouthEnhancer.OUTER_LIP], dtype=np.float32)

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
