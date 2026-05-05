import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from tqdm import tqdm
from utils import save_csv
from stats_utils import merge, save_fig, centroid_cosine_dist, bootstrap_ci
from analyse.descriptive import filter_canonical_vowels

F1_HZ = "F1_mid"
F2_HZ = "F2_mid"

ROPE_HZ_LO = -20.0
ROPE_HZ_HI = +20.0


def _classify_rope(ci_lo, ci_hi, rope_lo, rope_hi):
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "undetermined"
    if ci_hi < rope_lo or ci_lo > rope_hi:
        return "non-equivalent"
    if ci_lo >= rope_lo and ci_hi <= rope_hi:
        return "equivalent"
    return "indeterminate"


def _acoustic_lme_cis(vowels, out, n_boot=2000):
    print(f"\n  Acoustic CIs via speaker-level bootstrap "
          f"({n_boot} resamples) and LME for interaction")

    df = vowels.copy()
    df["is_L2"]   = (df["l1_status"] == "L2").astype(int)
    df["is_male"] = (df["gender"].str.lower() == "m").astype(int)

    rows = []
    phones = sorted(df["phoneme"].unique())

    for ph in phones:
        sub = df[df["phoneme"] == ph]
        spk_l1 = sub.groupby("speaker_id")["l1_status"].first()
        if spk_l1.value_counts().min() < 2:
            continue

        for response, label in [(F1_HZ, "F1"), (F2_HZ, "F2")]:
            d = sub[[response, "l1_status", "speaker_id"]].dropna()
            if d["speaker_id"].nunique() < 4 or len(d) < 20:
                continue

            def _diff(dd, _resp=response):
                l1 = dd[dd["l1_status"] == "L1"][_resp].values
                l2 = dd[dd["l1_status"] == "L2"][_resp].values
                if len(l1) == 0 or len(l2) == 0:
                    return np.nan
                return float(l1.mean() - l2.mean())

            point = _diff(d)
            ci_lo, ci_hi = bootstrap_ci(d, _diff, n_boot=n_boot, verbose=False)

            rows.append(dict(
                phoneme=ph, feature=label, contrast="L1_vs_L2",
                coef=point, ci_lo=ci_lo, ci_hi=ci_hi,
                rope_lo=ROPE_HZ_LO, rope_hi=ROPE_HZ_HI,
                rope_class=_classify_rope(ci_lo, ci_hi, ROPE_HZ_LO, ROPE_HZ_HI),
                p=np.nan,
            ))

            d_full = sub[[response, "is_L2", "is_male", "speaker_id"]].dropna()
            try:
                m_full = smf.mixedlm(f"{response} ~ is_L2 * is_male",
                                     d_full, groups=d_full["speaker_id"]).fit(
                    reml=True, method=["lbfgs", "powell"])
                ci = m_full.conf_int()
                if "is_L2:is_male" in m_full.fe_params.index:
                    coef = float(m_full.fe_params["is_L2:is_male"])
                    lo = float(ci.loc["is_L2:is_male", 0])
                    hi = float(ci.loc["is_L2:is_male", 1])
                    rows.append(dict(
                        phoneme=ph, feature=label, contrast="L1xGender",
                        coef=coef, ci_lo=lo, ci_hi=hi,
                        rope_lo=ROPE_HZ_LO, rope_hi=ROPE_HZ_HI,
                        rope_class=_classify_rope(lo, hi, ROPE_HZ_LO, ROPE_HZ_HI),
                        p=float(m_full.pvalues["is_L2:is_male"]),
                    ))
            except Exception as exc:
                print(f"    [{ph}/{label}] interaction LME failed: {exc}")

    df_out = pd.DataFrame(rows)
    save_csv(out / "tables" / "8_1_acoustic_ci_rope.csv", df_out, index=False)
    return df_out


def _neural_bootstrap_cis(vowels, layers, out, n_boot=2000):
    print(f"\n  Neural bootstrap CIs ({n_boot} resamples, speaker-level)")
    rows = []
    rng = np.random.default_rng(42)

    for lyr in tqdm(layers, desc="    layers"):
        pair = lyr.get("pca_clust")
        if pair is None:
            print(f"    [WARN] {lyr['name']}: no pca_clust, skipping")
            continue
        ids, vecs = pair
        df = merge(vowels, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]

        intra_dists = []
        for spk, sg in df.groupby("speaker_id"):
            for ph, pg in sg.groupby("phoneme"):
                if len(pg) >= 2:
                    V = pg[pc_cols].values.astype(np.float32)
                    nrm = np.linalg.norm(V, axis=1, keepdims=True)
                    nrm[nrm == 0] = 1
                    Vn = V / nrm
                    cs = Vn @ Vn.T
                    cd = 1 - cs[np.triu_indices(len(V), k=1)]
                    intra_dists.extend(cd.tolist())
        delta = float(np.nanmean(intra_dists)) if intra_dists else np.nan

        for ph in tqdm(sorted(df["phoneme"].unique()),
                       desc=f"    {lyr['name']} phonemes", leave=False):
            sub = df[df["phoneme"] == ph]
            spk_l1 = sub.groupby("speaker_id")["l1_status"].first()
            if spk_l1.value_counts().min() < 2:
                continue
            if (sub["l1_status"] == "L1").sum() < 3 or \
               (sub["l1_status"] == "L2").sum() < 3:
                continue

            speakers = spk_l1.index.tolist()
            spk_vecs = {}
            spk_label = {}
            for spk in speakers:
                sg = sub[sub["speaker_id"] == spk]
                spk_vecs[spk] = sg[pc_cols].values.astype(np.float32)
                spk_label[spk] = sg["l1_status"].iloc[0]

            l1_arr = np.vstack([spk_vecs[s] for s in speakers if spk_label[s] == "L1"])
            l2_arr = np.vstack([spk_vecs[s] for s in speakers if spk_label[s] == "L2"])
            point = centroid_cosine_dist(l1_arr, l2_arr)

            n_spk = len(speakers)
            speakers_arr = np.array(speakers)
            label_arr = np.array([spk_label[s] for s in speakers])
            boot = np.empty(n_boot)

            for b in range(n_boot):
                idx = rng.integers(0, n_spk, size=n_spk)
                sampled_spk = speakers_arr[idx]
                sampled_lbl = label_arr[idx]

                l1_parts = [spk_vecs[s] for s, lb in zip(sampled_spk, sampled_lbl)
                            if lb == "L1"]
                l2_parts = [spk_vecs[s] for s, lb in zip(sampled_spk, sampled_lbl)
                            if lb == "L2"]
                if not l1_parts or not l2_parts:
                    boot[b] = np.nan
                    continue
                v1 = np.vstack(l1_parts)
                v2 = np.vstack(l2_parts)
                boot[b] = centroid_cosine_dist(v1, v2)

            ci_lo = float(np.nanpercentile(boot, 2.5))
            ci_hi = float(np.nanpercentile(boot, 97.5))
            rope_class = _classify_rope(ci_lo, ci_hi, 0.0, delta)

            rows.append(dict(
                layer=lyr["name"], phoneme=ph,
                point=point, ci_lo=ci_lo, ci_hi=ci_hi,
                rope_lo=0.0, rope_hi=delta,
                rope_class=rope_class,
            ))

    df_out = pd.DataFrame(rows)
    save_csv(out / "tables" / "8_2_neural_ci_rope.csv", df_out, index=False)
    return df_out


def _forest_plot(df, fig_path, x_label, title, rope_lo=None, rope_hi=None):
    if df.empty:
        return
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    colors = {"equivalent": "green", "non-equivalent": "red",
              "indeterminate": "orange", "undetermined": "grey"}

    for k, r in df.iterrows():
        c = colors.get(r["rope_class"], "grey")
        ax.barh(k, r["ci_hi"] - r["ci_lo"],
                left=r["ci_lo"], height=0.4, color=c, alpha=0.5)
        point = r.get("coef", r.get("point", np.nan))
        ax.plot([point], [k], "k|", markersize=10)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    if rope_lo is not None and rope_hi is not None:
        ax.axvspan(rope_lo, rope_hi, alpha=0.15, color="blue",
                   label=f"ROPE [{rope_lo:.3g}, {rope_hi:.3g}]")
        ax.legend(loc="best")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["phoneme"].values, fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    save_fig(fig, fig_path)


def _make_forest_plots(ac_rows, neural_rows, out):
    print("\n   Forest plots")

    sub = ac_rows[(ac_rows["feature"] == "F1") &
                  (ac_rows["contrast"] == "L1_vs_L2")]
    _forest_plot(sub, out / "figures" / "8_forest_acoustic_F1_L1vsL2.png",
                 "L1 − L2 difference (F1, Hz)",
                 "L1/L2 contrast on F1 — LME estimates with ±20 Hz ROPE",
                 rope_lo=ROPE_HZ_LO, rope_hi=ROPE_HZ_HI)

    sub = ac_rows[(ac_rows["feature"] == "F2") &
                  (ac_rows["contrast"] == "L1_vs_L2")]
    _forest_plot(sub, out / "figures" / "8_forest_acoustic_F2_L1vsL2.png",
                 "L1 − L2 difference (F2, Hz)",
                 "L1/L2 contrast on F2 — LME estimates with ±20 Hz ROPE",
                 rope_lo=ROPE_HZ_LO, rope_hi=ROPE_HZ_HI)

    sub = ac_rows[(ac_rows["feature"] == "F1") &
                  (ac_rows["contrast"] == "L1xGender")]
    _forest_plot(sub, out / "figures" / "8_forest_acoustic_F1_interaction.png",
                 "L1 × Gender interaction coef (F1, Hz)",
                 "L1 × Gender interaction on F1 — LME estimates with ±20 Hz ROPE",
                 rope_lo=ROPE_HZ_LO, rope_hi=ROPE_HZ_HI)

    sub = ac_rows[(ac_rows["feature"] == "F2") &
                  (ac_rows["contrast"] == "L1xGender")]
    _forest_plot(sub, out / "figures" / "8_forest_acoustic_F2_interaction.png",
                 "L1 × Gender interaction coef (F2, Hz)",
                 "L1 × Gender interaction on F2 — LME estimates with ±20 Hz ROPE",
                 rope_lo=ROPE_HZ_LO, rope_hi=ROPE_HZ_HI)

    for layer in neural_rows["layer"].unique():
        sub = neural_rows[neural_rows["layer"] == layer].reset_index(drop=True)
        if sub.empty:
            continue
        delta = float(sub["rope_hi"].iloc[0])
        _forest_plot(sub,
                     out / "figures" / f"8_forest_neural_{layer}.png",
                     "Cosine distance between L1 and L2 centroids",
                     f"L1/L2 contrast in {layer} — bootstrap CIs with ROPE [0, δ]",
                     rope_lo=0.0, rope_hi=delta)


def _summary_table(ac_rows, neural_rows, out):
    summary = []

    for _, r in ac_rows.iterrows():
        summary.append(dict(
            phoneme=r["phoneme"],
            representation=f"acoustic_{r['feature']}_{r['contrast']}",
            point=r["coef"],
            ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
            rope_class=r["rope_class"],
        ))
    for _, r in neural_rows.iterrows():
        summary.append(dict(
            phoneme=r["phoneme"],
            representation=r["layer"],
            point=r["point"],
            ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
            rope_class=r["rope_class"],
        ))

    save_csv(out / "tables" / "8_4_rope_summary.csv",
             pd.DataFrame(summary), index=False)


def ci_rope(ac, layers, out):
    print("\n=== CIs and ROPE ===")
    vowels = filter_canonical_vowels(ac, verbose=False)

    if F1_HZ not in vowels.columns or F2_HZ not in vowels.columns:
        print("  Hz formants missing, skipping ROPE")
        return

    ac_rows = _acoustic_lme_cis(vowels, out)
    neural_rows = _neural_bootstrap_cis(vowels, layers, out, n_boot=2000)

    _make_forest_plots(ac_rows, neural_rows, out)
    _summary_table(ac_rows, neural_rows, out)

    print("ROPE done")
