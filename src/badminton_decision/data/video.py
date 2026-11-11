from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image


class VideoFrameReader:
    def __init__(self, num_frames: int = 8, duration_seconds: float = 3.0, cache_size: int = 8, jitter_seconds: float = 0.0):
        self.num_frames = int(num_frames)
        self.duration_seconds = float(duration_seconds)
        self.cache_size = int(cache_size)
        self.jitter_seconds = float(jitter_seconds)
        self.cache: OrderedDict[tuple, list[Image.Image]] = OrderedDict()

    def read(self, video_path: str | Path, target_frame: float, fps: float, training: bool = False) -> list[Image.Image]:
        path = str(Path(video_path).resolve())
        jitter = random.uniform(-self.jitter_seconds, self.jitter_seconds) if training and self.jitter_seconds > 0 else 0.0
        key = (path, round(float(target_frame), 1), round(float(fps), 3), round(jitter, 2), self.num_frames, self.duration_seconds)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")
        fps = float(fps) if fps and float(fps) > 0 else 30.0
        center_sec = float(target_frame) / fps + jitter
        half = self.duration_seconds / 2.0
        start_sec = max(0.0, center_sec - half)
        times = np.linspace(start_sec, start_sec + self.duration_seconds, self.num_frames, endpoint=False)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps and native_fps > 0:
            # The annotation FPS is authoritative when present; fallback only if invalid.
            pass
        images = []
        last_frame = None
        for timestamp in times:
            frame_index = max(0, int(round(timestamp * fps)))
            if total_frames > 0:
                frame_index = min(frame_index, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_index - 1, total_frames - 1)))
                    ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    raise RuntimeError(f"Failed to decode frame {frame_index} from {path}")
            last_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        cap.release()
        self.cache[key] = images
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return images
