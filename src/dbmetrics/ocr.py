"""Dense local text detection; model files must already exist."""
from ._ocr import Limits, Frame, PaddleDetector, measure_clip, run_benchmark

__all__ = ["Limits", "Frame", "PaddleDetector", "measure_clip", "run_benchmark"]
