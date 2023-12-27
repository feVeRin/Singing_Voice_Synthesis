import numpy as np
import torch.nn.functional as F

def median_nan(a):
    return np.median(a[~np.isnan(a)])


def padding(sound_stft):
    n_frames = sound_stft.size(-1)
    n_pad = (64 - n_frames % 64) % 64
    if n_pad:
        left = n_pad // 2
        right = n_pad - left
        return F.pad(sound_stft, (left, right), mode='reflect'), (left, right)
    else:
        return sound_stft, (0, 0)
