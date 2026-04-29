from pathlib import Path
import pandas as pd
import re
import parselmouth
from parselmouth.praat import call
import argparse
from tqdm import tqdm
from utils.py import save_csv


def load_metadata(path_dir, filename):
    path = path_dir / filename
    metadata = pd.read_csv(path, sep=";")

    metadata.columns = metadata.columns\
        .str.strip()\
        .str.lower()\
        .str.replace(" ", "_")

    metadata["spk"] = metadata["spk"].str.strip().str.lower()

    metadata["l1"] = metadata["l1"]\
        .str.strip()\
        .str.lower()\
        .map({"fr": "L1", "ru": "L2"})

    metadata["gender"] = metadata["gender"].str.strip()
    metadata = metadata.set_index("spk")
    return metadata


def map_repetitions(path_dir, filename):
    path = path_dir / filename
    rep = pd.read_csv(path, sep="\t")
    mapping: dict[int, tuple[str, int]] = {}
    occ_cols = [c for c in rep.columns if c.startswith("occ.")]

    for _, row in rep.iterrows():
        word = str(row["Word"]).strip()
        for col in occ_cols:
            rep_idx = int(col.split(".")[1])
            val = row[col]
            corp_num = int(val)
            mapping[corp_num] = (word, rep_idx)
    return mapping  # (13: (word, rep))


def parse_wav_name(stem):
    stem = stem.strip().lower()
    match = re.search(r"frcorp(\d+)$", stem)
    corp_num = int(match.group(1)) if match else None
    spk = stem.split("_")[0]
    return {"speaker_id": spk, "corp_num": corp_num}


def iter_phonemes(textgrid, phoneme_tier):
    n_intervals = call(textgrid, "Get number of intervals", phoneme_tier)

    for i in range(1, n_intervals + 1):
        label = call(textgrid, "Get label of interval", phoneme_tier, i).strip()

        if not label or label.lower().startswith("ding"):
            continue

        onset = call(textgrid, "Get start time of interval", phoneme_tier, i)
        offset = call(textgrid, "Get end time of interval", phoneme_tier, i)

        duration_ms = (offset - onset) * 1000.0
        if duration_ms <= 0:
            continue

        yield label, onset, offset, duration_ms


def parse_file(wav_path, metadata, corpus_index, phoneme_tier):
    info = parse_wav_name(wav_path.stem)
    spk = info["speaker_id"]
    corp_num = info["corp_num"]

    if corp_num is None or corp_num not in corpus_index:
        return []
    if spk not in metadata.index:
        return []

    word, rep_idx = corpus_index[corp_num]

    if "distractor" in word.lower():
        return []

    spk_meta = metadata.loc[spk]

    textgrid_path = wav_path.with_suffix(".TextGrid")
    if not textgrid_path.exists():
        return []

    textgrid = parselmouth.read(str(textgrid_path))

    rows = []
    for position, (label, onset, offset, duration_ms) in enumerate(
            iter_phonemes(textgrid, phoneme_tier)):

        speaker_id = spk.upper()
        rows.append({
            "phoneme_id":  f"{speaker_id}_{word}_rep{rep_idx}_p{position}_{label}",
            "speaker_id":  speaker_id,
            "sentence_id": word,
            "repetition":  rep_idx,
            "phoneme":     label,
            "onset_s":     round(onset, 6),
            "offset_s":    round(offset, 6),
            "duration_ms": round(duration_ms, 3),
            "l1_status":   spk_meta["l1"],
            "gender":      spk_meta["gender"],
            "wav_path":    str(wav_path.resolve()),
            "tg_path":     str(textgrid_path.resolve()),
        })

    return rows


def parse_corpus(raw_dir, metadata_file, corr_file, phoneme_tier, output_path):
    raw_dir = Path(raw_dir)
    tg_dir = raw_dir / "wav_et_textgrids" / "FRcorp_textgrids_only"

    metadata = load_metadata(raw_dir, metadata_file)
    corpus_index = map_repetitions(raw_dir, corr_file)

    wav_files = sorted(tg_dir.rglob("*.wav"))

    all_rows = []
    for wav_path in tqdm(wav_files, desc="Parsing corpus"):
        all_rows.extend(
            parse_file(wav_path, metadata, corpus_index, phoneme_tier)
        )

    df = pd.DataFrame(all_rows, columns=[
        "phoneme_id", "speaker_id", "sentence_id", "repetition",
        "phoneme", "onset_s", "offset_s", "duration_ms",
        "l1_status", "gender", "wav_path", "tg_path",
    ])

    save_csv(output_path, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",       required=True)
    parser.add_argument("--metadata_file", default="metadata_RUFR.csv")
    parser.add_argument("--corr_file",     default="RUFRcorr.csv")
    parser.add_argument("--phoneme_tier",  type=int, default=2)
    parser.add_argument("--output",        default="outputs/table.csv")
    args = parser.parse_args()

    parse_corpus(raw_dir=args.raw_dir,
                 metadata_file=args.metadata_file,
                 corr_file=args.corr_file,
                 phoneme_tier=args.phoneme_tier,
                 output_path=args.output)
