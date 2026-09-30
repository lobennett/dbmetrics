"""Offline video measurements; importing this package loads no models."""

from .cpu import METRICS, measure_video

__version__ = '0.2.0'
__all__ = ['METRICS', 'measure_video']
