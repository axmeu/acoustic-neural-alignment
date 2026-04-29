import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import argparse
from tqdm import tqdm
from utils import save_csv, is_vowel


def get_max_formant(gender):
    return 5000.0 if gender == "f" else 4500.0


def extract_formants(sound, time, max_formant, n_formants=5):
    formant = call(sound, "To Formant (burg)",
                   0.0,
                   n_formants,
                   max_formant,
                   0.025,
                   50.0)
    results = {}
    for k, name in enumerate(("F1", "F2", "F3"), start=1):
        val = call(formant, "Get value at time", k, time, "Hertz", "Linear")
        results[name] = val if val == val else np.nan
    return results


def extract_f0(sound, time):
    try:
        pitch = call(sound, "To Pitch (ac)", 0.0, 75, 600)

        if not call(pitch, "Is voiced at time", time, "Hertz"):
            return np.nan

        val = call(pitch, "Get value at time", time, "Hertz", "Linear")
        return val if val == val else np.nan
    except Exception:
        return np.nan


def extract_scg(sound):
    try:
        spectrum = call(sound, "To Spectrum", "yes")
        return call(spectrum, "Get centre of gravity", 2.0)
    except Exception:
        return np.nan


def extract_token(row, sound, n_formants=5):
    segment = sound.extract_part(
        from_time=row["onset_s"],
        to_time=row["offset_s"],
        preserve_times=True,
    )

    midpoint = (row["onset_s"] + row["offset_s"]) / 2.0
    max_formant = get_max_formant(row["gender"])

    formants_mid = extract_formants(segment, midpoint, max_formant, n_formants)
    f0 = extract_f0(segment, midpoint)
    scg = extract_scg(segment)

    is_v = is_vowel(row["phoneme"])
    if is_v:
        F3 = formants_mid["F3"]
    else:
        F3 = np.nan

    feats = {
        "phoneme_id": row["phoneme_id"],
        "phoneme":    row["phoneme"],
        "F1_mid":     formants_mid["F1"],
        "F2_mid":     formants_mid["F2"],
        "F3_mid":     F3,
        "f0_mid":     f0,
        "SCG":        scg,
    }

    if row["duration_ms"] > 80.0:
        t25 = row["onset_s"] + (row["offset_s"] - row["onset_s"]) * 0.25
        t75 = row["onset_s"] + (row["offset_s"] - row["onset_s"]) * 0.75

        for t, tag in [(t25, "25"), (t75, "75")]:
            f = extract_formants(segment, t, max_formant, n_formants)
            feats[f"F1_{tag}"] = f["F1"]
            feats[f"F2_{tag}"] = f["F2"]
    else:
        for tag in ("25", "75"):
            feats[f"F1_{tag}"] = np.nan
            feats[f"F2_{tag}"] = np.nan

    return feats


def empty_feats(phoneme_id):
    return {
        "phoneme_id": phoneme_id,
        "F1_mid": np.nan, "F2_mid": np.nan, "F3_mid": np.nan,
        "f0_mid": np.nan, "SCG":    np.nan,
        "F1_25":  np.nan, "F2_25":  np.nan,
        "F1_75":  np.nan, "F2_75":  np.nan,
    }


def report_missing(df):
    cols = ["F1_mid", "F2_mid", "F3_mid", "f0_mid", "SCG"]
    print("\nMissing rates:")
    for c in cols:
        if c in df.columns:
            print(c, df[c].isna().mean())

    print("\nBy phoneme:")
    print(df.groupby("phoneme")[cols].apply(lambda x: x.isna().mean()))


def extract_acoustics(table_path, output_path, n_formants=5):
    df = pd.read_csv(table_path)
    records = []
    for wav_path, group in tqdm(df.groupby("wav_path"), desc="Extracting acoustics"):
        try:
            sound = parselmouth.Sound(wav_path)
        except Exception as e:
            print(f"[WARN] could not load {wav_path}: {e}")
            for _, row in group.iterrows():
                records.append(empty_feats(row["phoneme_id"]))
            continue

        for _, row in group.iterrows():
            try:
                feats = extract_token(row, sound, n_formants=n_formants)
            except Exception as e:
                print(f"[WARN] {row['phoneme_id']}: {e}")
                feats = empty_feats(row["phoneme_id"])
            records.append(feats)

    out = pd.DataFrame(records, columns=[
        "phoneme_id",
        "F1_mid", "F2_mid", "F3_mid",
        "f0_m