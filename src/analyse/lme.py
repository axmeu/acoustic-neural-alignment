import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from tqdm import tqdm
from utils import save_csv
from stats_utils import merge
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from analyse.descriptive import (
    F1_LOB, F2_LOB, FRENCH_ORAL_VOWELS,
    filter_canonical_vowels,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered")


VOWEL_HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",  "ə": "mid",
    "ɛ": "mid",
    "a": "low",  "ɑ": "low",
}


def _fit_null(df, response, use_reml):
    return smf.mixedlm(f"{response} ~ 1", df, groups=df["speaker_id"]) \
        .fit(reml=use_reml, method="powell")


def _fit_main(df, response):
    return smf.mixedlm(f"{response} ~ is_L2 + is_male",
                       df, groups=df["speaker_id"]) \
        .fit(reml=False, method="powell")


def _fit_full(df, response):
    return smf.mixedlm(f"{response} ~ is_L2 * is_male",
                       df, groups=df["speaker_id"]) \
        .fit(reml=False, method="powell")


def _fit_extended(df, response):
    if df["height"].nunique() < 2:
        return None
    return smf.mixedlm(f"{response} ~ is_L2 * is_male + C(height)",
                       df, groups=df["speaker_id"]) \
        .fit(reml=False, method="lbfgs")


def _fit_random_slope(df, response):
    formula = (f"{response} ~ is_L2 * is_male + C(height)"
               if df["height"].nunique() >= 2
               else f"{response} ~ is_L2 * is_male")
    return smf.mixedlm(formula, df, groups=df["speaker_id"],
                       re_formula="~1 + is_L2") \
        .fit(reml=False, method="powell")


def _compute_icc(model_reml):
    var_u = float(model_reml.cov_re.iloc[0, 0])
    var_e = float(model_reml.scale)
    return var_u / (var_u + var_e) if (var_u + var_e) > 0 else np.nan


def _compute_r2(model):
    try:
        beta = model.fe_params.values
        X = model.model.exog
        var_f = float(np.var(X @ beta, ddof=1))
        var_u = float(model.cov_re.iloc[0, 0]) if model.cov_re.size else 0.0
        var_e = float(model.scale)
        denom = var_f + var_u + var_e
        if denom <= 0:
            return np.nan, np.nan
        return var_f / denom, (var_f + var_u) / denom
    except Exception:
        return np.nan, np.nan


def _lrt(m_simple, m_complex):
    df_diff = (m_complex.model.exog.shape[1] - m_simple.model.exog.shape[1])
    if df_diff <= 0:
        return np.nan, np.nan, np.nan
    lr = 2 * (m_complex.llf - m_simple.llf)
    p = stats.chi2.sf(lr, df=df_diff)
    return float(lr), int(df_diff), float(p)


def _fit_all_models(df, response, tag, has_height):
    df = df[[response, "is_L2", "is_male", "height", "speaker_id"]].dropna()
    if df["speaker_id"].nunique() < 4 or len(df) < 20:
        return []

    rows = []

    try:
        m_null_reml = _fit_null(df, response, use_reml=True)
        icc = _compute_icc(m_null_reml)
        r2m, r2c = _compute_r2(m_null_reml)
        print(f"    [{tag}] null OK: icc={icc:.3f}, r2m={r2m:.3f}")
        rows.append(dict(tag=tag, model="null_model",
                         aic=m_null_reml.aic, bic=m_null_reml.bic,
                         icc=icc, r2_marginal=r2m, r2_conditional=r2c))
    except Exception as exc:
        print(f"    [{tag}] null/REML failed: {exc}")
        return rows

    try:
        m_null_ml = _fit_null(df, response, use_reml=False)
    except Exception as exc:
        print(f"    [{tag}] null/ML failed: {exc}")
        return rows

    try:
        m_main = _fit_main(df, response)
        r2m, r2c = _compute_r2(m_main)
        lr, df_lr, p_lr = _lrt(m_null_ml, m_main)
        rows.append(dict(tag=tag, model="main",
                         aic=m_main.aic, bic=m_main.bic,
                         r2_marginal=r2m, r2_conditional=r2c,
                         lrt_vs_previous=lr, lrt_df=df_lr, lrt_p=p_lr))
    except Exception as exc:
        print(f"    [{tag}] main failed: {exc}")
        m_main = None

    try:
        m_full = _fit_full(df, response)
        r2m, r2c = _compute_r2(m_full)
        if m_main is not None:
            lr, df_lr, p_lr = _lrt(m_main, m_full)
        else:
            lr, df_lr, p_lr = (np.nan, np.nan, np.nan)
        rows.append(dict(tag=tag, model="full",
                         aic=m_full.aic, bic=m_full.bic,
                         r2_marginal=r2m, r2_conditional=r2c,
                         lrt_vs_previous=lr, lrt_df=df_lr, lrt_p=p_lr))

        if "is_L2:is_male" in m_full.fe_params.index:
            ci = m_full.conf_int().loc["is_L2:is_male"]
            rows.append(dict(tag=tag, model="full_interaction_coef",
                             coef=float(m_full.fe_params["is_L2:is_male"]),
                             p=float(m_full.pvalues["is_L2:is_male"]),
                             ci_lo=float(ci[0]), ci_hi=float(ci[1])))
    except Exception as exc:
        print(f"    [{tag}] full failed: {exc}")
        m_full = None

    m_extended = None
    if has_height and m_full is not None:
        try:
            m_extended = _fit_extended(df, response)
            if m_extended is not None:
                r2m, r2c = _compute_r2(m_extended)
                lr, df_lr, p_lr = _lrt(m_full, m_extended)
                rows.append(dict(tag=tag, model="extended",
                                 aic=m_extended.aic, bic=m_extended.bic,
                                 r2_marginal=r2m, r2_conditional=r2c,
                                 lrt_vs_previous=lr, lrt_df=df_lr, lrt_p=p_lr))
        except Exception as exc:
            print(f"    [{tag}] extended failed: {exc}")
    else:
        rows.append(dict(tag=tag, model="extended",
                         note="skipped: vowel height constant in subset"))

    base_for_rs = m_extended if m_extended is not None else m_full
    if base_for_rs is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # capture ConvergenceWarning
                m_rs = _fit_random_slope(df, response)
            r2m, r2c = _compute_r2(m_rs)
            lr, df_lr, p_lr = _lrt(base_for_rs, m_rs)
            var_slope = (float(m_rs.cov_re.iloc[1, 1])
                         if m_rs.cov_re.shape[0] >= 2 else np.nan)
            rows.append(dict(tag=tag, model="random_slope",
                             aic=m_rs.aic, bic=m_rs.bic,
                             r2_marginal=r2m, r2_conditional=r2c,
                             lrt_vs_previous=lr, lrt_df=df_lr, lrt_p=p_lr,
                             var_slope=var_slope, converged=True))
        except Exception as exc:
            rows.append(dict(tag=tag, model="random_slope",
                             converged=False,
                             note=f"degenerate fit (expected): {type(exc).__name__}"))

    return rows


def lme(ac, layers, out):
    print("\n=== Linear Mixed-Effects Models ===")
    vowels = filter_canonical_vowels(ac, verbose=False)

    if F1_LOB not in vowels.columns:
        print("  F1_mid_lob missing, skipping LME")
        return

    vowels = vowels.copy()
    vowels["is_L2"] = (vowels["l1_status"] == "L2").astype(int)
    vowels["is_male"] = (vowels["gender"].str.lower() == "m").astype(int)
    vowels["height"] = vowels["phoneme"].map(VOWEL_HEIGHT)

    target_phonemes = sorted([p for p in FRENCH_ORAL_VOWELS
                              if p in vowels["phoneme"].values])
    print(f"  Phonemes: {target_phonemes}")
    print(f"  L2 speakers: {vowels.groupby('speaker_id')['is_L2']
          .first()
          .sum()}/{vowels['speaker_id'].nunique()}")

    rows = []

    print("\n   Acoustic LME, per-phoneme")
    for ph in tqdm(target_phonemes, desc="    phonemes"):
        sub = vowels[vowels["phoneme"] == ph]
        rows.extend(_fit_all_models(sub, "F1_mid",
                                    f"acoustic_F1_{ph}", has_height=False))
        rows.extend(_fit_all_models(sub, "F2_mid",
                                    f"acoustic_F2_{ph}", has_height=False))

    print("\n   Acoustic LME, across phonemes (extended model uses height)")
    rows.extend(_fit_all_models(vowels, "F1_mid",
                                "acoustic_F1_all", has_height=True))
    rows.extend(_fit_all_models(vowels, "F2_mid",
                                "acoustic_F2_all", has_height=True))

    print("\n   Neural LME, per-phoneme on PC1..PC5")
    for lyr in tqdm(layers, desc="    layers"):
        pair = lyr.get("pca_lme")
        if pair is None:
            print(f"    [WARN] {lyr['name']}: no pca_lme, skipping")
            continue
        ids, vecs = pair
        df = merge(vowels, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")][:5]

        for ph in target_phonemes:
            sub = df[df["phoneme"] == ph]
            if len(sub) < 20:
                continue
            for dim in pc_cols:
                rows.extend(_fit_all_models(sub, dim,
                                            f"{lyr['name']}_{dim}_{ph}",
                                            has_height=False))

        for dim in pc_cols:
            rows.extend(_fit_all_models(df, dim,
                                        f"{lyr['name']}_{dim}_all",
                                        has_height=True))

    save_csv(out / "tables" / "7_lme_results.csv",
             pd.DataFrame(rows), index=True)
    print("LME done")
