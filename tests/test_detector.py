"""Tests for the detection layer (motion + ensemble + IoU)."""

from __future__ import annotations

import numpy as np
import pytest

from macbook.detector import (
    Detection,
    EnsembleDetector,
    MotionDetector,
    _bbox_iou,
)


class TestDetection:
    def test_aim_point_below_bbox(self):
        det = Detection(bbox=(100, 100, 200, 300), centroid=(150, 200), confidence=0.8, source="motion")
        ax, ay = det.aim_point(drop_fraction=0.15)
        assert ax == 150
        assert ay == 300 + int(0.15 * 200)

    def test_aim_point_zero_drop_at_bottom(self):
        det = Detection(bbox=(0, 0, 100, 100), centroid=(50, 50), confidence=0.5, source="motion")
        assert det.aim_point(drop_fraction=0.0) == (50, 100)

    def test_geometry_helpers(self):
        det = Detection(bbox=(10, 20, 110, 220), centroid=(60, 120), confidence=0.5, source="motion")
        assert det.width == 100
        assert det.height == 200
        assert det.area == 20_000


class TestBboxIoU:
    def test_identical_boxes_iou_one(self):
        b = (0, 0, 100, 100)
        assert _bbox_iou(b, b) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        assert _bbox_iou((0, 0, 50, 50), (200, 200, 250, 250)) == 0.0

    def test_half_overlap(self):
        a = (0, 0, 100, 100)
        b = (50, 0, 150, 100)
        # Intersection = 50*100 = 5000; Union = 100*100 + 100*100 - 5000 = 15000
        assert _bbox_iou(a, b) == pytest.approx(5000 / 15000)


class TestMotionDetector:
    def test_first_frame_no_detection(self):
        d = MotionDetector()
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        assert d.detect(frame) is None

    def test_static_scene_no_detection(self):
        d = MotionDetector(min_area=200)
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        for _ in range(10):
            assert d.detect(frame) is None

    def test_moving_blob_detected(self, moving_blob_frames):
        d = MotionDetector(min_area=200)
        results = [d.detect(f) for f in moving_blob_frames]
        non_none = [r for r in results if r is not None]
        assert len(non_none) >= 5
        last = non_none[-1]
        assert last.source == "motion"
        x1, y1, x2, y2 = last.bbox
        assert x2 > x1 and y2 > y1
        assert 100 < (x1 + x2) / 2 < 640

    def test_min_area_filter_rejects_tiny_motion(self):
        import cv2
        d = MotionDetector(min_area=50_000)
        frames = []
        for i in range(15):
            f = np.full((480, 640, 3), 60, dtype=np.uint8)
            if i > 4:
                cv2.circle(f, (200 + i * 5, 200), 4, (220, 220, 220), -1)
            frames.append(f)
        results = [d.detect(f) for f in frames]
        assert all(r is None for r in results)


class TestEnsembleDetector:
    def test_motion_only_returns_motion(self, moving_blob_frames):
        ens = EnsembleDetector(motion=MotionDetector(min_area=200), yolo=None)
        last_det = None
        for f in moving_blob_frames:
            r = ens.detect(f)
            if r is not None:
                last_det = r
        assert last_det is not None
        assert last_det.source == "motion"

    def test_no_detection_when_static(self):
        ens = EnsembleDetector(motion=MotionDetector(min_area=200), yolo=None)
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        for _ in range(5):
            assert ens.detect(frame) is None

    def test_ensemble_fusion_when_yolo_overlaps(self):
        """Synthetic: motion sees the moving box, fake YOLO sees an overlapping box.

        Box must MOVE every frame — a static box becomes background under
        MOG2 after a few frames and motion detection stops firing.
        """
        import cv2
        m = MotionDetector(min_area=100)

        class _FakeYolo:
            def detect(self, frame):
                return Detection(
                    bbox=(245, 195, 365, 305),
                    centroid=(305, 250),
                    confidence=0.8,
                    source="yolo",
                )

        ens = EnsembleDetector(motion=m, yolo=_FakeYolo(), iou_match=0.1)
        frames = []
        for i in range(20):
            f = np.full((480, 640, 3), 60, dtype=np.uint8)
            if i > 4:
                x = 100 + i * 12
                cv2.rectangle(f, (x - 50, 200), (x + 50, 300), (220, 220, 220), -1)
            frames.append(f)
        last = None
        for f in frames:
            r = ens.detect(f)
            if r:
                last = r
        assert last is not None
        assert last.source == "ensemble"
        assert last.confidence > 0.5

    def test_yolo_only_when_motion_quiet(self):
        class _FakeYolo:
            def detect(self, frame):
                return Detection(
                    bbox=(10, 20, 30, 40),
                    centroid=(20, 30),
                    confidence=0.7,
                    source="yolo",
                )

        ens = EnsembleDetector(motion=MotionDetector(min_area=10_000_000), yolo=_FakeYolo())
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        det = ens.detect(frame)
        assert det is not None
        assert det.source == "yolo"
