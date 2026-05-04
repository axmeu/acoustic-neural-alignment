import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import argparse
from tqdm import tqdm
from utils import save_csv, is_vowel, is_fricative


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


def extract_f0(sound, onset, offset):
    try:
        pitch = call(sound, "To Pitch (ac)",
                     0.0,    # time step (0 = auto)
                     75.0,   # pitch floor (Hz)
                     15,     # max number of candidates
                     "yes",  # very accurate
                     0.03,   # silence threshold
                     0.45,   # voicing threshold
                     0.01,   # octave cost
                     0.35,   # octave-jump cost
                     0.14,   # voiced/unvoiced cost
                     600.0)  # pitch ceiling (Hz)
        val = call(pitch, "Get mean", onset, offset, "Hertz")
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
    f0 = extract_f0(segment, row["onset_s"], row["offset_s"])
    scg = extract_scg(segment) if is_fricative(row["phoneme"]) else np.nan
    F3 = formants_mid["F3"] if is_vowel(row["phoneme"]) else np.nan

    feats = {
        "phoneme_id": row["phoneme_id"],
        "phoneme":    row["phoneme"],
        "F1_mid":     formants_mid["F1"],
        "F2_mid":     formants_mid["F2"],
        "F3_mid":     F3,
        "f0_mean":    f0,
        "SCG":        scg
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
        "f0_mean": np.nan, "SCG":    np.nan,
        "F1_25":  np.nan, "F2_25":  np.nan,
        "F1_75":  np.nan, "F2_75":  np.nan,
    }


def report_missing(df):
    cols = ["F1_mid", "F2_mid", "F3_mid", "f0_mean", "SCG"]
    print("\nMissing rates:")
    for c in cols:
        if c in df.columns:
            rate = df[c].isna().mean()
            print(f"  {c}: {rate:.2%}")

    print("\nMissing rates by phoneme:")
    by_phoneme = (
        df.groupby("phoneme")[cols]
        .apply(lambda x: x.isna().mean())
    )
    print(by_phoneme.to_string())

    print("\nMissing rates by group (l1_status, gender):")
    by_group = (
        df.groupby(["l1_status", "gender"])[cols]
        .apply(lambda x: x.isna().mean())
    )
    print(by_group.to_string())


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
        "f0_mean", "SCG",
        "F1_25", "F2_25",
        "F1_75", "F2_75"
    ])

    meta_cols = ["phoneme_id", "phoneme", "speaker_id",
                 "sentence_id", "repetition", "duration_ms",
                 "l1_status", "gender"]
    out = df[meta_cols].merge(out, on="phoneme_id", how="left")
    print("out shape:", out.shape)
    print("out columns:", out.columns.tolist())
    print(out[["phoneme_id", "F1_mid"]].head(10))

    report_missing(out)
    save_csv(output_path, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table",     required=True)
    parser.add_argument("--output",    default="outputs/features_acoustic.csv")
    parser.add_argument("--n_formants", type=int, default=5)
    args = parser.parse_args()

    extract_acoustics(
        table_path=args.table,
        output_path=args.output,
        n_formants=args.n_formants,
    )
