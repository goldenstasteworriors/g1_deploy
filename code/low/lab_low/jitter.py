"""GigaWorld IMU loss weighting, with explicitly identified orientation proxy.

Reference: gwp-imu-jitter-ablation-20260821/scripts/augment_sonic_imu.py
and world_action_model/trainer/wa_casual_trainer_mot.py.
"""
import numpy as np
import torch
import torch.nn.functional as F


def orientation_speed(quaternions, timestamps):
    """Causal base-orientation angular speed [rad/s], NOT a camera gyro reading."""
    q = np.asarray(quaternions, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape != (len(t), 4) or not np.isfinite(q).all() or (norm < 1e-8).any():
        raise ValueError("Invalid root orientation (wxyz)")
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError("IMU timestamps must strictly increase")
    q = q / norm
    # |dot| makes q and -q identical; first frame has no previous reading.
    angle = 2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(-1)), 0, 1))
    return np.r_[0., angle / dt].astype(np.float32)


def latent_jitter(scores, latent_frames, temporal_factor=4):
    """Align clean frame + non-overlapping temporal VAE groups with max pooling."""
    if scores.ndim != 2 or not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Jitter scores must be finite, nonnegative [B,T]")
    if scores.shape[1] != 1 + (latent_frames - 1) * temporal_factor:
        raise ValueError("Jitter/video latent timeline mismatch")
    if latent_frames == 1:
        return scores
    tail = F.max_pool1d(scores[:, None, 1:].float(), temporal_factor, temporal_factor)[:, 0]
    return torch.cat([scores[:, :1], tail], dim=1)


def jitter_weights(scores, scale, floor):
    if scale <= 0 or not 0 < floor <= 1:
        raise ValueError("Require scale > 0 and 0 < floor <= 1")
    return floor + (1 - floor) / (1 + (scores.float() / scale).square())
