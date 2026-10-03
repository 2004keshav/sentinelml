"""
Drift detection for SentinelML.
Compares live/incoming transaction batches against the REAL reference
sample (data/reference_sample.csv) using Population Stability Index (PSI).
Never uses synthetic/randomly generated reference data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-6  # avoids log(0) / divide-by-zero in the PSI formula
N_BINS = 10      # decile bins -> standard PSI convention

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class DriftDetector:
    def __init__(self, reference_path: str | Path | None = None):
        if reference_path is None:
            # path relative to this file -> works in Colab AND in Docker,
            # no hardcoded /content/drive path
            reference_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "reference_sample.csv"
            )
        self.reference_path = Path(reference_path)
        self.reference_df = pd.read_csv(self.reference_path)
        self._bin_edges = self._compute_bin_edges()

    def _compute_bin_edges(self) -> dict[str, np.ndarray]:
        edges = {}
        for col in FEATURE_COLUMNS:
            quantiles = np.linspace(0, 1, N_BINS + 1)
            bin_edges = np.quantile(self.reference_df[col], quantiles)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            # unique() guards against constant/near-constant features
            # producing duplicate bin edges
            edges[col] = np.unique(bin_edges)
        return edges

    @staticmethod
    def _bin_percentages(series: pd.Series, edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(series, bins=edges)
        total = max(len(series), 1)
        return counts / total

    def calculate_psi(self, feature: str, live_series: pd.Series) -> float:
        edges = self._bin_edges[feature]
        ref_pct = self._bin_percentages(self.reference_df[feature], edges)
        live_pct = self._bin_percentages(live_series, edges)
        ref_pct = np.clip(ref_pct, EPSILON, None)
        live_pct = np.clip(live_pct, EPSILON, None)
        psi = np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct))
        return float(psi)

    @staticmethod
    def _status(psi: float) -> str:
        if psi < 0.1:
            return "stable"
        if psi < 0.25:
            return "moderate_shift"
        return "significant_drift"

    def compute_drift_report(self, live_df: pd.DataFrame) -> dict:
        per_feature = {}
        for col in FEATURE_COLUMNS:
            if col not in live_df.columns:
                continue
            psi = self.calculate_psi(col, live_df[col])
            per_feature[col] = {"psi": round(psi, 4), "status": self._status(psi)}

        overall_psi = float(np.mean([v["psi"] for v in per_feature.values()]))
        return {
            "n_live_rows": len(live_df),
            "n_reference_rows": len(self.reference_df),
            "overall_psi": round(overall_psi, 4),
            "overall_status": self._status(overall_psi),
            "per_feature": per_feature,
        }
