"""Risk-relevant structure detectors (coordination burst, propagation, external channel, delegation)."""
from __future__ import annotations

from .base import Detection
from .coordination import BurstBaseline, detect_coordination_burst
from .delegation import detect_emergent_delegation
from .external_channel import detect_external_channel
from .propagation import detect_propagation


def run_all(episode, events, baseline: "BurstBaseline") -> list[Detection]:
    """Run every detector on one episode and attach results to it."""
    dets = [
        detect_coordination_burst(episode, baseline),
        detect_propagation(episode, events),
        detect_external_channel(episode, events),
        detect_emergent_delegation(episode, events),
    ]
    episode.detections = dets
    return dets


__all__ = ["Detection", "BurstBaseline", "run_all", "detect_coordination_burst", "detect_propagation",
           "detect_external_channel", "detect_emergent_delegation"]
