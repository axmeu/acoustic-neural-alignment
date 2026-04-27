import soundfile as sf
import resampy
import numpy as np
import tempfile
import os
from pathlib import Path


def load_audio(wav_path, target_sr=16_000):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def get_frame_indices(onset_s, offset_s, n_frames, audio_duration_s):
    frame_dur = audio_duration_s / n_frames
    t_start = int(np.floor(onset_s / frame_dur))
    t_end = int(np.ceil(offset_s / frame_dur))
    return max(0, t_start), min(n_frames, t_end)


def save_npz(output_path, ids_arr, vecs_arr):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".npz")
    os.close(tmp_fd)
    try:
        np.savez(tmp_path, ids=ids_arr, vectors=vecs_arr)
        os.replace(tmp_path, output_path)
        print(f"\nSaved {vecs_arr.shape[0]} vectors of dim {vecs_arr.shape[1]} in {output_path}")
    except Exception:
        os.unlink(tmp_path)
        raise
