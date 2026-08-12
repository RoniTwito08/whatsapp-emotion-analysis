"""statistical_analysis.py

Paired statistical comparison of E2 (prefix-aware AlephBERT, text only) vs
E3 (prefix-aware AlephBERT + 26 pure behavioral features).

Tasks:
  1. Verify prediction files are paired correctly
  2. Paired bootstrap (10,000 samples) for Macro-F1 delta with 95% CI
  3. McNemar exact test on correctness disagreements
  4. Error analysis: fixes, regressions, shared errors
  5. Confidence (probability) comparison across outcome groups
  6. Publication-quality figures

Usage:
    cd text-model
    python statistical_analysis.py

All inputs come from existing E2/E3 prediction CSVs and E1 comparison JSON.
Nothing is retrained. No model is modified. No test data is used for training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent
E2_DIR      = BASE_DIR / "outputs" / "early_detection_e2"
E3_DIR      = BASE_DIR / "outputs" / "early_detection_e3"
E1_JSON     = BASE_DIR / "outputs" / "early_detection_e1" / "comparison_table.json"
OUT_DIR     = BASE_DIR / "outputs" / "statistical_analysis"

FRACTIONS   = [0.25, 0.50, 0.75, 1.00]
FRAC_LABELS = ["25%", "50%", "75%", "100%"]
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

# Known E1 macro-F1 (from comparison_table.json — verified against training logs)
E1_MACRO_F1 = {0.25: 0.3328, 0.50: 0.3377, 0.75: 0.5114, 1.00: 0.9804}


# ---------------------------------------------------------------------------
# Task 1 — Verify predictions
# ---------------------------------------------------------------------------

def load_predictions(e2_dir: Path, e3_dir: Path, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load E2 and E3 prediction CSVs for a given fraction."""
    tag = f"{int(frac * 100):03d}pct"
    e2 = pd.read_csv(e2_dir / f"predictions_{tag}.csv")
    e3 = pd.read_csv(e3_dir / f"predictions_{tag}.csv")
    return e2, e3


def verify_predictions(e2: pd.DataFrame, e3: pd.DataFrame, frac: float) -> None:
    """Raise ValueError if E2/E3 predictions are not correctly paired."""
    tag = f"{int(frac*100)}%"
    errors: List[str] = []

    if len(e2) != 459:
        errors.append(f"E2 {tag}: expected 459 rows, got {len(e2)}")
    if len(e3) != 459:
        errors.append(f"E3 {tag}: expected 459 rows, got {len(e3)}")

    if e2["conversation_id"].duplicated().any():
        errors.append(f"E2 {tag}: duplicate conversation IDs found")
    if e3["conversation_id"].duplicated().any():
        errors.append(f"E3 {tag}: duplicate conversation IDs found")

    if len(e2) == len(e3):
        if not (e2["conversation_id"].values == e3["conversation_id"].values).all():
            # Check if at least the sets match
            if set(e2["conversation_id"]) != set(e3["conversation_id"]):
                errors.append(f"{tag}: E2 and E3 have different conversation ID sets")
            else:
                errors.append(f"{tag}: conversation IDs are the same set but in different order — analysis requires identical ordering")
        if not (e2["actual_label"].values == e3["actual_label"].values).all():
            errors.append(f"{tag}: E2 and E3 have different true labels for the same rows")

    if errors:
        raise ValueError("Prediction verification FAILED:\n  " + "\n  ".join(errors))

    print(f"  [{tag}] Verification PASSED: 459 paired conversations, labels match.")


# ---------------------------------------------------------------------------
# Task 2 — Paired bootstrap
# ---------------------------------------------------------------------------

def _macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary macro F1 with numpy (fast, no sklearn overhead)."""
    f1s = []
    for cls in (0, 1):
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def paired_bootstrap(
    y_true: np.ndarray,
    y_e2: np.ndarray,
    y_e3: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Paired bootstrap for Macro-F1 delta (E3 - E2).

    For each resample, the same indices are used for both models so the
    comparison is always within the same conversations.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta_samples = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f2 = _macro_f1_binary(y_true[idx], y_e2[idx])
        f3 = _macro_f1_binary(y_true[idx], y_e3[idx])
        delta_samples[i] = f3 - f2

    obs_e2    = _macro_f1_binary(y_true, y_e2)
    obs_e3    = _macro_f1_binary(y_true, y_e3)
    obs_delta = obs_e3 - obs_e2
    ci_lo, ci_hi = np.percentile(delta_samples, [2.5, 97.5])
    prop_e3_gt_e2 = float((delta_samples > 0).mean())

    return {
        "observed_e2_macro_f1":     round(obs_e2,    4),
        "observed_e3_macro_f1":     round(obs_e3,    4),
        "observed_delta":           round(obs_delta,  4),
        "bootstrap_mean_delta":     round(float(delta_samples.mean()), 4),
        "ci_95_lo":                 round(float(ci_lo),  4),
        "ci_95_hi":                 round(float(ci_hi),  4),
        "ci_95_lo_pp":              round(float(ci_lo) * 100, 2),
        "ci_95_hi_pp":              round(float(ci_hi) * 100, 2),
        "prop_e3_gt_e2":            round(prop_e3_gt_e2, 4),
        "n_bootstrap":              n_bootstrap,
        "bootstrap_seed":           seed,
        "_delta_samples":           delta_samples,  # kept for plotting, not serialised
    }


# ---------------------------------------------------------------------------
# Task 3 — McNemar exact test
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> Dict[str, Any]:
    """Two-sided McNemar exact test.

    b = E2 correct, E3 wrong (regressions)
    c = E2 wrong,  E3 correct (fixes)

    Under H0: b/(b+c) = 0.5 (no systematic difference).
    Uses exact binomial test.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_disagreements": 0,
                "p_value": 1.0, "statistic_chi2": None,
                "significant_at_0.05": False,
                "note": "No disagreements — models make identical predictions"}

    result = binomtest(min(b, c), n, 0.5, alternative="two-sided")
    p = float(result.pvalue)

    # Chi-square with continuity correction (for reporting, valid when n > 25)
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else None

    return {
        "b_e2_right_e3_wrong": b,
        "c_e2_wrong_e3_right": c,
        "n_disagreements": n,
        "p_value": round(p, 6),
        "chi2_with_continuity": round(chi2, 4) if chi2 is not None else None,
        "significant_at_0.05": p < 0.05,
        "significant_at_0.01": p < 0.01,
        "note": "Exact two-sided McNemar via scipy.stats.binomtest",
    }


# ---------------------------------------------------------------------------
# Task 4 — Error analysis
# ---------------------------------------------------------------------------

def error_analysis(e2: pd.DataFrame, e3: pd.DataFrame) -> Dict[str, Any]:
    """Identify fixes, regressions, and shared errors."""
    e2_correct = e2["correct"].astype(int).values
    e3_correct = e3["correct"].astype(int).values
    cids       = e2["conversation_id"].values
    y_true     = e2["actual_label"].values

    fixes:       List[Dict] = []  # E2 wrong, E3 correct
    regressions: List[Dict] = []  # E2 correct, E3 wrong
    both_wrong:  List[Dict] = []  # both wrong
    both_right: int = 0

    for i, cid in enumerate(cids):
        entry = {
            "conversation_id": cid,
            "actual_label":    y_true[i],
            "e2_predicted":    e2.iloc[i]["predicted_label"],
            "e3_predicted":    e3.iloc[i]["predicted_label"],
        }
        if e2_correct[i] and e3_correct[i]:
            both_right += 1
        elif not e2_correct[i] and e3_correct[i]:
            fixes.append(entry)
        elif e2_correct[i] and not e3_correct[i]:
            regressions.append(entry)
        else:
            both_wrong.append(entry)

    return {
        "fixes":       fixes,
        "regressions": regressions,
        "both_wrong":  both_wrong,
        "both_right_count": both_right,
        "n_fixes":       len(fixes),
        "n_regressions": len(regressions),
        "n_both_wrong":  len(both_wrong),
        "net_corrected": len(fixes) - len(regressions),
    }


# ---------------------------------------------------------------------------
# Task 5 — Confidence analysis
# ---------------------------------------------------------------------------

def confidence_analysis(
    e2: pd.DataFrame,
    e3: pd.DataFrame,
    error_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare prediction confidence (max probability) across outcome groups."""
    fix_ids  = {d["conversation_id"] for d in error_results["fixes"]}
    reg_ids  = {d["conversation_id"] for d in error_results["regressions"]}

    def _conf(df: pd.DataFrame) -> pd.Series:
        return df[["probability_interested", "probability_losing_interest"]].max(axis=1)

    results: Dict[str, Any] = {}
    for name, df in [("e2", e2), ("e3", e3)]:
        conf = _conf(df)
        correct_mask = df["correct"].astype(bool)
        fix_mask     = df["conversation_id"].isin(fix_ids)
        reg_mask     = df["conversation_id"].isin(reg_ids)

        results[name] = {
            "mean_conf_correct":    round(float(conf[correct_mask].mean()),  4) if correct_mask.any() else None,
            "mean_conf_incorrect":  round(float(conf[~correct_mask].mean()), 4) if (~correct_mask).any() else None,
            "mean_conf_fixes":      round(float(conf[fix_mask].mean()),      4) if fix_mask.any() else None,
            "mean_conf_regressions":round(float(conf[reg_mask].mean()),      4) if reg_mask.any() else None,
        }
    return results


# ---------------------------------------------------------------------------
# Task 6 — Figures
# ---------------------------------------------------------------------------

PALETTE = {"E1": "#9e9e9e", "E2": "#1f77b4", "E3": "#d62728"}
MARKERS = {"E1": "s", "E2": "o", "E3": "^"}
X_TICKS = [25, 50, 75, 100]


def plot_e1_e2_e3(all_macro_f1: Dict[str, Dict[float, float]], out_path: Path) -> None:
    """Figure 1: Macro-F1 by prefix for E1, E2, E3."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model in ["E1", "E2", "E3"]:
        vals = all_macro_f1[model]
        xs   = [int(f * 100) for f in FRACTIONS]
        ys   = [vals[f] for f in FRACTIONS]
        ax.plot(xs, ys, marker=MARKERS[model], color=PALETTE[model],
                linewidth=2, markersize=7, label=model)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=PALETTE[model])

    ax.set_xlabel("Conversation progress (% of messages available)", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Early Detection: Macro-F1 by Prefix\n"
                 "E1=full-conv model, E2=prefix-aware text, E3=E2+behavioral",
                 fontsize=10)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([f"{x}%" for x in X_TICKS])
    ax.set_ylim(0.25, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_delta_with_ci(bootstrap_by_frac: Dict[float, Dict], out_path: Path) -> None:
    """Figure 2: E3 - E2 Macro-F1 delta with 95% bootstrap CI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    xs       = [int(f * 100) for f in FRACTIONS]
    deltas   = [bootstrap_by_frac[f]["observed_delta"] * 100 for f in FRACTIONS]
    ci_lo    = [(bootstrap_by_frac[f]["observed_delta"] - bootstrap_by_frac[f]["ci_95_lo"]) * 100
                for f in FRACTIONS]
    ci_hi    = [(bootstrap_by_frac[f]["ci_95_hi"] - bootstrap_by_frac[f]["observed_delta"]) * 100
                for f in FRACTIONS]
    colors   = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]

    ax.bar(xs, deltas, color=colors, alpha=0.7, width=8, zorder=3)
    ax.errorbar(xs, deltas, yerr=[ci_lo, ci_hi], fmt="none",
                ecolor="black", capsize=5, linewidth=1.5, zorder=4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for x, d, lo, hi in zip(xs, deltas, ci_lo, ci_hi):
        sign = "+" if d >= 0 else ""
        ax.annotate(f"{sign}{d:.2f}pp", (x, d + (hi + 0.3) * (1 if d >= 0 else -1)),
                    ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Conversation progress (% of messages available)", fontsize=11)
    ax.set_ylabel("Macro F1 delta: E3 − E2 (pp)", fontsize=11)
    ax.set_title("E3 vs E2: Macro-F1 Improvement with 95% Bootstrap CI\n"
                 "Green=E3 better, Red=E2 better", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}%" for x in xs])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Statistical Analysis: E2 vs E3 on Early Detection Test Set")
    print("=" * 68)

    # Load E1 data for figures
    e1_data: Dict[str, float] = {}
    if E1_JSON.exists():
        raw = json.loads(E1_JSON.read_text())
        e1_data = {f: raw.get(str(int(f * 100)), {}).get("macro_f1", np.nan)
                   for f in FRACTIONS}

    all_bootstrap:   Dict[float, Dict] = {}
    all_mcnemar:     Dict[float, Dict] = {}
    all_error:       Dict[float, Dict] = {}
    all_confidence:  Dict[float, Dict] = {}
    all_macro_f1: Dict[str, Dict[float, float]] = {
        "E1": e1_data,
        "E2": {},
        "E3": {},
    }

    # ── TASK 1: Verify ─────────────────────────────────────────────────────
    print("\n[Task 1] Verifying prediction file alignment...")
    all_e2, all_e3 = {}, {}
    for frac in FRACTIONS:
        e2, e3 = load_predictions(E2_DIR, E3_DIR, frac)
        verify_predictions(e2, e3, frac)  # raises on mismatch
        all_e2[frac] = e2
        all_e3[frac] = e3

    # ── TASK 2: Paired bootstrap ─────────────────────────────────────────
    print("\n[Task 2] Paired bootstrap (N={:,}, seed={})...".format(N_BOOTSTRAP, BOOTSTRAP_SEED))
    for frac in FRACTIONS:
        e2, e3 = all_e2[frac], all_e3[frac]
        y_true = (e2["actual_label"] == "losing_interest").astype(int).values
        y_e2   = (e2["predicted_label"] == "losing_interest").astype(int).values
        y_e3   = (e3["predicted_label"] == "losing_interest").astype(int).values

        bs = paired_bootstrap(y_true, y_e2, y_e3)
        all_bootstrap[frac] = bs
        all_macro_f1["E2"][frac] = bs["observed_e2_macro_f1"]
        all_macro_f1["E3"][frac] = bs["observed_e3_macro_f1"]

        ci_sign = ("+" if bs["ci_95_lo"] >= 0 else
                   "~" if bs["ci_95_lo"] < 0 < bs["ci_95_hi"] else "-")
        print(f"  [{int(frac*100):3d}%] obs_delta={bs['observed_delta']*100:+.2f}pp  "
              f"95%CI=[{bs['ci_95_lo_pp']:+.2f}, {bs['ci_95_hi_pp']:+.2f}]pp  "
              f"P(E3>E2)={bs['prop_e3_gt_e2']:.3f}  "
              f"CI_sign={ci_sign}")

    # ── TASK 3: McNemar ──────────────────────────────────────────────────
    print("\n[Task 3] McNemar exact test...")
    for frac in FRACTIONS:
        e2, e3 = all_e2[frac], all_e3[frac]
        err    = error_analysis(e2, e3)
        b      = err["n_regressions"]   # E2 right, E3 wrong
        c      = err["n_fixes"]         # E2 wrong, E3 right
        mc     = mcnemar_exact(b, c)
        all_mcnemar[frac] = mc
        sig    = "*** p<.01" if mc["significant_at_0.01"] else ("* p<.05" if mc["significant_at_0.05"] else "n.s.")
        print(f"  [{int(frac*100):3d}%] b(E2✓E3✗)={b:3d}  c(E2✗E3✓)={c:3d}  "
              f"n={mc['n_disagreements']:3d}  p={mc['p_value']:.4f}  {sig}")

    # ── TASK 4: Error analysis ────────────────────────────────────────────
    print("\n[Task 4] Error analysis...")
    for frac in FRACTIONS:
        e2, e3 = all_e2[frac], all_e3[frac]
        err    = error_analysis(e2, e3)
        all_error[frac] = err
        print(f"  [{int(frac*100):3d}%] fixes={err['n_fixes']:3d}  "
              f"regressions={err['n_regressions']:3d}  "
              f"both_wrong={err['n_both_wrong']:3d}  "
              f"net={err['net_corrected']:+d}")

    # ── TASK 5: Confidence analysis ───────────────────────────────────────
    print("\n[Task 5] Confidence analysis...")
    for frac in FRACTIONS:
        e2, e3 = all_e2[frac], all_e3[frac]
        conf   = confidence_analysis(e2, e3, all_error[frac])
        all_confidence[frac] = conf
        print(f"  [{int(frac*100):3d}%] E2 conf correct={conf['e2']['mean_conf_correct']}  "
              f"incorrect={conf['e2']['mean_conf_incorrect']} | "
              f"E3 correct={conf['e3']['mean_conf_correct']}  "
              f"incorrect={conf['e3']['mean_conf_incorrect']}")

    # ── TASK 6: Figures ───────────────────────────────────────────────────
    print("\n[Task 6] Generating figures...")
    plot_e1_e2_e3(all_macro_f1, OUT_DIR / "fig1_macro_f1_by_prefix.png")
    plot_delta_with_ci(all_bootstrap, OUT_DIR / "fig2_delta_with_ci.png")

    # ── Serialise results ─────────────────────────────────────────────────
    _save_results(OUT_DIR, all_bootstrap, all_mcnemar, all_error, all_confidence)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("SUMMARY — E2 vs E3 (bootstrap 95% CI, McNemar p-value)")
    print("=" * 68)
    print(f"{'Prefix':>8}  {'E2 mF1':>8}  {'E3 mF1':>8}  {'Δ(pp)':>8}  "
          f"{'95% CI (pp)':>20}  {'P(E3>E2)':>10}  {'McNemar p':>11}  {'Sig':>5}")
    print("-" * 87)
    for frac in FRACTIONS:
        bs = all_bootstrap[frac]
        mc = all_mcnemar[frac]
        ci = f"[{bs['ci_95_lo_pp']:+.2f}, {bs['ci_95_hi_pp']:+.2f}]"
        sig = ("p<.01" if mc["significant_at_0.01"] else
               "p<.05" if mc["significant_at_0.05"] else "n.s.")
        sign = "+" if bs["observed_delta"] > 0 else ""
        print(f"  {int(frac*100):>5}%  "
              f"{bs['observed_e2_macro_f1']:>8.4f}  "
              f"{bs['observed_e3_macro_f1']:>8.4f}  "
              f"{sign}{bs['observed_delta']*100:>7.2f}  "
              f"{ci:>20}  "
              f"{bs['prop_e3_gt_e2']:>10.3f}  "
              f"{mc['p_value']:>11.4f}  {sig:>5}")

    _write_summary(OUT_DIR, all_bootstrap, all_mcnemar, all_error)

    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _save_results(
    out: Path,
    bootstrap: Dict[float, Dict],
    mcnemar:   Dict[float, Dict],
    error:     Dict[float, Dict],
    confidence: Dict[float, Dict],
) -> None:
    # statistical_results.json — master JSON
    master: Dict[str, Any] = {}
    for frac in FRACTIONS:
        key = f"{int(frac*100)}pct"
        bs  = {k: v for k, v in bootstrap[frac].items() if k != "_delta_samples"}
        master[key] = {
            "bootstrap": bs,
            "mcnemar":   mcnemar[frac],
            "error_summary": {k: v for k, v in error[frac].items()
                              if not isinstance(v, list)},
            "confidence": confidence[frac],
        }
    with (out / "statistical_results.json").open("w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2, ensure_ascii=False)

    # bootstrap_results.csv
    rows = []
    for frac in FRACTIONS:
        bs = bootstrap[frac]
        mc = mcnemar[frac]
        rows.append({
            "prefix_pct":           int(frac * 100),
            "e2_macro_f1":          bs["observed_e2_macro_f1"],
            "e3_macro_f1":          bs["observed_e3_macro_f1"],
            "observed_delta_pp":    round(bs["observed_delta"] * 100, 3),
            "bootstrap_mean_delta_pp": round(bs["bootstrap_mean_delta"] * 100, 3),
            "ci_95_lo_pp":          bs["ci_95_lo_pp"],
            "ci_95_hi_pp":          bs["ci_95_hi_pp"],
            "prop_e3_gt_e2":        bs["prop_e3_gt_e2"],
            "n_bootstrap":          bs["n_bootstrap"],
            "mcnemar_p":            mc["p_value"],
            "sig_0.05":             mc["significant_at_0.05"],
        })
    pd.DataFrame(rows).to_csv(out / "bootstrap_results.csv", index=False)

    # mcnemar_results.csv
    mc_rows = []
    for frac in FRACTIONS:
        mc = mcnemar[frac]
        mc_rows.append({"prefix_pct": int(frac * 100), **mc})
    pd.DataFrame(mc_rows).to_csv(out / "mcnemar_results.csv", index=False)

    # error_analysis.csv — per-conversation
    err_rows = []
    for frac in FRACTIONS:
        for outcome, label in [("fixes", "fix"), ("regressions", "regression"),
                                ("both_wrong", "both_wrong")]:
            for entry in error[frac][outcome]:
                err_rows.append({
                    "prefix_pct":       int(frac * 100),
                    "outcome":          label,
                    "conversation_id":  entry["conversation_id"],
                    "actual_label":     entry["actual_label"],
                    "e2_predicted":     entry["e2_predicted"],
                    "e3_predicted":     entry["e3_predicted"],
                })
    pd.DataFrame(err_rows).to_csv(out / "error_analysis.csv", index=False)

    print("\n  Saved: statistical_results.json, bootstrap_results.csv, "
          "mcnemar_results.csv, error_analysis.csv")


def _write_summary(
    out: Path,
    bootstrap: Dict[float, Dict],
    mcnemar:   Dict[float, Dict],
    error:     Dict[float, Dict],
) -> None:
    lines = [
        "Statistical Comparison: E2 (text-only) vs E3 (text + behavioral)",
        "=" * 68,
        f"Test set: 459 conversations (same split for all experiments)",
        f"Bootstrap: {N_BOOTSTRAP:,} samples, seed={BOOTSTRAP_SEED}",
        f"McNemar: two-sided exact test (scipy.stats.binomtest)",
        "",
        "Macro-F1 results, confidence intervals, and significance:",
        "",
    ]
    for frac in FRACTIONS:
        bs  = bootstrap[frac]
        mc  = mcnemar[frac]
        err = error[frac]
        ci_covers_zero = bs["ci_95_lo"] < 0 < bs["ci_95_hi"]
        sig = ("p<0.01, statistically significant" if mc["significant_at_0.01"] else
               "p<0.05, statistically significant" if mc["significant_at_0.05"] else
               "p≥0.05, not statistically significant")

        lines += [
            f"Prefix {int(frac*100)}%:",
            f"  E2 macro F1:   {bs['observed_e2_macro_f1']:.4f}",
            f"  E3 macro F1:   {bs['observed_e3_macro_f1']:.4f}",
            f"  Delta:         {bs['observed_delta']*100:+.2f} pp",
            f"  95% bootstrap CI: [{bs['ci_95_lo_pp']:+.2f}, {bs['ci_95_hi_pp']:+.2f}] pp",
            f"  CI covers zero:   {'YES → inconclusive' if ci_covers_zero else 'NO → directional evidence'}",
            f"  P(E3 > E2):    {bs['prop_e3_gt_e2']:.3f} (of bootstrap samples)",
            f"  McNemar:       fixes={err['n_fixes']}, regressions={err['n_regressions']}, {sig}",
            f"  Net corrected: {err['net_corrected']:+d} conversations",
            "",
        ]

    # Conservative conclusions
    lines += [
        "=" * 68,
        "CONSERVATIVE RESEARCH CONCLUSIONS",
        "=" * 68,
        "",
    ]
    for frac in FRACTIONS:
        bs  = bootstrap[frac]
        mc  = mcnemar[frac]
        ci_lo = bs["ci_95_lo"]
        ci_hi = bs["ci_95_hi"]
        pct   = int(frac * 100)
        delta = bs["observed_delta"] * 100

        if mc["significant_at_0.05"] and ci_lo > 0:
            conclusion = (
                f"E3 shows a statistically significant improvement at {pct}% prefix "
                f"(McNemar p={mc['p_value']:.4f}, 95% CI entirely above zero). "
                f"The +{delta:.2f}pp gain has statistical support."
            )
        elif mc["significant_at_0.05"] and ci_lo < 0:
            conclusion = (
                f"McNemar test is significant at {pct}% (p={mc['p_value']:.4f}), "
                f"but the bootstrap 95% CI crosses zero — the direction is uncertain."
            )
        elif not mc["significant_at_0.05"] and delta > 0:
            conclusion = (
                f"At {pct}%, E3 shows a positive delta ({delta:+.2f}pp) but this "
                f"does not reach statistical significance (McNemar p={mc['p_value']:.4f}). "
                f"The result is directionally positive but inconclusive."
            )
        elif not mc["significant_at_0.05"] and delta < 0:
            conclusion = (
                f"At {pct}%, E3 underperforms E2 by {delta:.2f}pp (McNemar p={mc['p_value']:.4f}, n.s.). "
                f"No reliable benefit from behavioral features at this prefix."
            )
        else:
            conclusion = (
                f"At {pct}%, no meaningful or significant difference between E2 and E3 "
                f"(delta={delta:+.2f}pp, McNemar p={mc['p_value']:.4f})."
            )
        lines.append(f"{pct}%: {conclusion}")
        lines.append("")

    lines += [
        "OVERALL:",
        "The evidence for behavioral features improving early detection is",
        "statistically mixed. Claims of improvement should be limited to",
        "prefixes with both significant McNemar tests AND CIs that exclude zero.",
        "Differences below ~1pp on a 459-sample test set should be treated",
        "as potentially arising from sampling variance unless confirmed with",
        "multiple seeds or held-out datasets.",
    ]

    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  Saved: summary.txt")


if __name__ == "__main__":
    run()
