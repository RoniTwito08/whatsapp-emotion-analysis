"""Tests for statistical_analysis.py — no model downloads required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from statistical_analysis import (
    _macro_f1_binary,
    error_analysis,
    mcnemar_exact,
    paired_bootstrap,
    verify_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preds(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic prediction dataframe."""
    rng = np.random.default_rng(seed)
    labels  = rng.integers(0, 2, n)
    preds   = rng.integers(0, 2, n)
    correct = (labels == preds).astype(int)
    label_names = ["interested", "losing_interest"]
    return pd.DataFrame({
        "conversation_id":         [f"c{i:03d}" for i in range(n)],
        "fraction":                [0.25] * n,
        "actual_label":            [label_names[l] for l in labels],
        "predicted_label":         [label_names[p] for p in preds],
        "probability_interested":  rng.random(n).round(4),
        "probability_losing_interest": rng.random(n).round(4),
        "correct":                 correct,
    })


def _exact_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic predictions with known correct count."""
    rng = np.random.default_rng(seed)
    n_correct  = 15
    n_wrong    = n - n_correct
    labels     = (["interested"] * 10 + ["losing_interest"] * 10)[:n]
    correct    = ([1] * n_correct + [0] * n_wrong)
    predicted  = []
    for i in range(n):
        if correct[i]:
            predicted.append(labels[i])
        else:
            predicted.append("interested" if labels[i] == "losing_interest" else "losing_interest")
    return pd.DataFrame({
        "conversation_id":             [f"c{i:03d}" for i in range(n)],
        "fraction":                    [0.5] * n,
        "actual_label":                labels,
        "predicted_label":             predicted,
        "probability_interested":      rng.random(n).round(4),
        "probability_losing_interest": rng.random(n).round(4),
        "correct":                     correct,
    })


# ---------------------------------------------------------------------------
# _macro_f1_binary
# ---------------------------------------------------------------------------

def test_macro_f1_perfect():
    y = np.array([0, 1, 0, 1])
    assert abs(_macro_f1_binary(y, y) - 1.0) < 1e-9


def test_macro_f1_all_wrong():
    y    = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0])
    assert abs(_macro_f1_binary(y, pred) - 0.0) < 1e-9


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng  = np.random.default_rng(7)
    y    = rng.integers(0, 2, 100)
    pred = rng.integers(0, 2, 100)
    sk   = f1_score(y, pred, average="macro", zero_division=0)
    mine = _macro_f1_binary(y, pred)
    assert abs(sk - mine) < 1e-9


# ---------------------------------------------------------------------------
# verify_predictions
# ---------------------------------------------------------------------------

def test_verify_passes_on_valid_input():
    df = _make_preds(459)
    # Should not raise
    verify_predictions(df.copy(), df.copy(), 0.25)


def test_verify_fails_on_wrong_count():
    df_good  = _make_preds(459)
    df_short = _make_preds(400)
    with pytest.raises(ValueError, match="459"):
        verify_predictions(df_short, df_good, 0.25)


def test_verify_fails_on_id_mismatch():
    df1 = _make_preds(459, seed=0)
    df2 = _make_preds(459, seed=1)  # different IDs
    # Both have 459 rows but different IDs → set mismatch or order mismatch
    # Inject a different conversation_id in df2
    df2 = df2.copy()
    df2.loc[0, "conversation_id"] = "DIFFERENT_ID_NOT_IN_DF1"
    with pytest.raises(ValueError):
        verify_predictions(df1, df2, 0.25)


def test_verify_fails_on_label_mismatch():
    df = _make_preds(459)
    df2 = df.copy()
    df2.loc[0, "actual_label"] = (
        "losing_interest" if df2.loc[0, "actual_label"] == "interested" else "interested"
    )
    with pytest.raises(ValueError, match="label"):
        verify_predictions(df, df2, 0.25)


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_returns_correct_keys():
    y = np.array([0, 1] * 10)
    bs = paired_bootstrap(y, y, y, n_bootstrap=100, seed=42)
    for key in ["observed_e2_macro_f1", "observed_e3_macro_f1", "observed_delta",
                "bootstrap_mean_delta", "ci_95_lo", "ci_95_hi", "prop_e3_gt_e2"]:
        assert key in bs, f"Missing key: {key}"


def test_bootstrap_identical_models_gives_zero_delta():
    """If E2 and E3 make identical predictions, delta must be 0."""
    y = np.array([0, 1] * 20)
    bs = paired_bootstrap(y, y, y, n_bootstrap=200, seed=42)
    assert abs(bs["observed_delta"]) < 1e-9
    assert abs(bs["bootstrap_mean_delta"]) < 1e-9
    # CI should straddle zero
    assert bs["ci_95_lo"] <= 0 <= bs["ci_95_hi"]


def test_bootstrap_deterministic():
    y    = np.array([0, 1, 0, 1, 0] * 20)
    pred = np.array([0, 1, 1, 0, 0] * 20)
    bs1  = paired_bootstrap(y, y, pred, n_bootstrap=500, seed=42)
    bs2  = paired_bootstrap(y, y, pred, n_bootstrap=500, seed=42)
    assert bs1["ci_95_lo"] == bs2["ci_95_lo"]
    assert bs1["ci_95_hi"] == bs2["ci_95_hi"]


def test_bootstrap_ci_is_finite_and_ordered():
    """CI lower bound must be <= observed delta <= upper bound."""
    y    = np.array([0, 1] * 50)
    pred = np.array([0, 1, 1, 0] * 25)
    bs   = paired_bootstrap(y, y, pred, n_bootstrap=500, seed=42)
    assert bs["ci_95_lo"] <= bs["observed_delta"] <= bs["ci_95_hi"], \
        "Observed delta must lie within the CI"


# ---------------------------------------------------------------------------
# mcnemar_exact
# ---------------------------------------------------------------------------

def test_mcnemar_zero_disagreements():
    mc = mcnemar_exact(0, 0)
    assert mc["p_value"] == 1.0
    assert not mc["significant_at_0.05"]


def test_mcnemar_balanced_disagreements_not_significant():
    """b = c → no evidence of systematic difference → p should be 1."""
    mc = mcnemar_exact(10, 10)
    assert mc["p_value"] == 1.0
    assert not mc["significant_at_0.05"]


def test_mcnemar_large_imbalance_is_significant():
    """Very large b vs c should be significant."""
    mc = mcnemar_exact(0, 50)
    assert mc["p_value"] < 0.001
    assert mc["significant_at_0.01"]


def test_mcnemar_symmetry():
    """Swapping b and c gives the same p-value (two-sided test)."""
    mc_ab = mcnemar_exact(5, 20)
    mc_ba = mcnemar_exact(20, 5)
    assert abs(mc_ab["p_value"] - mc_ba["p_value"]) < 1e-9


# ---------------------------------------------------------------------------
# error_analysis
# ---------------------------------------------------------------------------

def test_error_analysis_counts_sum_to_n():
    df = _exact_df(20)
    err = error_analysis(df, df)  # identical → no fixes, no regressions
    total = err["n_fixes"] + err["n_regressions"] + err["n_both_wrong"] + err["both_right_count"]
    assert total == 20


def test_error_analysis_identical_models():
    df = _exact_df(20)
    err = error_analysis(df, df)
    assert err["n_fixes"] == 0
    assert err["n_regressions"] == 0
    assert err["net_corrected"] == 0


def test_error_analysis_net_corrected():
    df_e2 = _make_preds(40, seed=0)
    df_e3 = df_e2.copy()
    # Force 5 improvements: turn 5 wrong E2 predictions into correct E3
    wrong_idx = df_e2.index[df_e2["correct"] == 0].tolist()[:5]
    for i in wrong_idx:
        df_e3.loc[i, "predicted_label"] = df_e3.loc[i, "actual_label"]
        df_e3.loc[i, "correct"] = 1
    err = error_analysis(df_e2, df_e3)
    assert err["n_fixes"] == 5
    assert err["net_corrected"] == 5


# ---------------------------------------------------------------------------
# File existence checks
# ---------------------------------------------------------------------------

def test_e2_e3_prediction_files_exist():
    base = Path(__file__).parent.parent / "outputs"
    for model in ["early_detection_e2", "early_detection_e3"]:
        for pct in ["025", "050", "075", "100"]:
            p = base / model / f"predictions_{pct}pct.csv"
            assert p.exists(), f"Missing: {p}"


def test_e1_comparison_json_exists():
    p = Path(__file__).parent.parent / "outputs" / "early_detection_e1" / "comparison_table.json"
    assert p.exists()
