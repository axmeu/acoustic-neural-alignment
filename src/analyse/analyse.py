import argparse
from pathlib import Path
import pandas as pd
from stats_utils import load_layer
from analyse import descriptive, statistical_tests, lme

SECTIONS = ["descriptive",  "tests", "lme"]


def load_data(acoustic_path, neural_root):
    ac_path = Path(acoustic_path)
    if not ac_path.exists():
        raise FileNotFoundError(f"Acoustic features not found: {ac_path}")

    ac = pd.read_csv(ac_path)
    print(f"Loaded acoustic: {len(ac)} tokens, "
          f"{ac['phoneme'].nunique()} phonemes, "
          f"{ac['speaker_id'].nunique()} speakers")

    layers = []
    root = Path(neural_root)
    if root.exists():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            lyr = load_layer(d)
            keys = [k for k, v in lyr.items() if k != "name" and v is not None]
            print(f"  Loaded layer {lyr['name']}: {keys}")
            layers.append(lyr)
    else:
        print(f"[WARN] Neural directory {root} does not exist")

    return ac, layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=SECTIONS + ["all"])
    parser.add_argument("--acoustic", default="outputs/features_acoustic_norm.csv")
    parser.add_argument("--neural-root", default="outputs/neural_norm")
    parser.add_argument("--output", default="outputs/analysis")
    args = parser.parse_args()

    out = Path(args.output)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    ac, layers = load_data(args.acoustic, args.neural_root)

    sections_to_run = SECTIONS if args.section == "all" else [args.section]

    if "descriptive" in sections_to_run:
        descriptive.desc_acoustic(ac, out)
        descriptive.desc_neural(ac, layers, out)
        descriptive.desc_rsm(ac, layers, out)

    if "tests" in sections_to_run:
        statistical_tests.group_comparisons(ac, layers, out)
        statistical_tests.distances(ac, layers, out)

    print(f"\nDone. Results written to: {out}")

    if "lme" in sections_to_run:
        lme.lme(ac, layers, out)


if __name__ == "__main__":
    main()
