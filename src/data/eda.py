#!/usr/bin/env python
"""
Exploratory Data Analysis (EDA) script for the **Richter's Predictor** dataset.

Usage (from project root):

    python src/data/eda.py --raw_dir data/raw --output_dir reports/eda

Outputs
-------
* CSV summaries -> <output_dir>/tables/
* PNG plots     -> <output_dir>/figures/
* feature_lists.json with cat_cols, geo_cols, num_cols

The script is *read‑only* for raw data; it never modifies files in
`data/raw/`.
"""
from __future__ import annotations

import argparse
import textwrap
import json
from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.switch_backend("agg")  # allow headless execution on CI/servers

###############################################################################
# Helper utilities
###############################################################################


def _ensure_dir(path: Path) -> None:
    """Crea directory se non esiste"""
    path.mkdir(parents=True, exist_ok=True)


def _human_path(path: Path) -> str:
    """Restituisce path se possibile"""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe as CSV and (optionally) Markdown."""
    csv_path = path.with_suffix(".csv")
    print(f"   saving table -> {_human_path(csv_path)}")
    df.to_csv(csv_path, index=True)
    try:
        md_path = path.with_suffix(".md")
        md_path.write_text(df.to_markdown(index=True), encoding='utf-8')
    except Exception:
        pass  # markdown export is optional


def _save_fig(fig: plt.Figure, path: Path) -> None:
    print(f"   saving figure -> {_human_path(path)}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


###############################################################################
# Main EDA
###############################################################################


def run_eda(raw_dir: Path, output_dir: Path) -> None:
    # 1. Carica raw CSVs
    print("\nLoading raw CSV files...")
    raw_train = raw_dir / "train_values.csv"
    raw_labels = raw_dir / "train_labels.csv"

    if not raw_train.exists() or not raw_labels.exists():
        raise FileNotFoundError(
            "train_values.csv or train_labels.csv not found in " f"{raw_dir}"
        )

    X_train = pd.read_csv(raw_train)
    y_train = pd.read_csv(raw_labels)
    df = X_train.merge(y_train, on="building_id", how="left")
    print(f"Rows in train_values: {len(X_train):,}; merged rows: {len(df):,}")

    # 2. Definisce gruppi di Features
    print("\nDefining feature groups...")
    cat_cols: List[str] = [
        "land_surface_condition",
        "foundation_type",
        "roof_type",
        "ground_floor_type",
        "other_floor_type",
        "position",
        "plan_configuration",
        "legal_ownership_status",
    ]
    geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
    num_cols = [
        c for c in X_train.columns if c not in cat_cols + geo_cols + ["building_id"]
    ]
    print(
        f"cat_cols: {len(cat_cols)}, geo_cols: {len(geo_cols)}, num_cols: {len(num_cols)}"
    )

    # 3. Prepara le cartelle di Output
    print("\nCreating output directories...")
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figures"
    _ensure_dir(tables_dir)
    _ensure_dir(figs_dir)

    # 4. Data types & Valori mancanti
    print("\nData-type overview & missing values...")
    dtypes = df.dtypes.to_frame(name="dtype")
    _save_table(dtypes, tables_dir / "data_types")

    miss_cols = df.isna().mean().mul(100).to_frame("missing_pct")
    _save_table(
        miss_cols.sort_values("missing_pct", ascending=False),
        tables_dir / "missing_columns",
    )

    miss_rows = df.isna().mean(axis=1).mul(100).to_frame("missing_pct")
    _save_table(
        miss_rows.describe(percentiles=[0.95, 0.99]),
        tables_dir / "missing_rows_summary",
    )

    # 5. Sommario & skewness
    print("\nNumerical summary & skewness...")
    num_desc = df[num_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    skew = df[num_cols].skew().round(3).to_frame("skew")
    _save_table(num_desc.join(skew), tables_dir / "numeric_summary")

    # 6. Conteggio categoriche
    print("\nCategorical value counts... (this may take a minute)")
    cat_counts_dir = tables_dir / "categorical_counts"
    _ensure_dir(cat_counts_dir)
    for col in cat_cols + geo_cols:
        print(f"   processing {col}...")
        counts = (
            df[col]
            .value_counts(dropna=False)
            .to_frame("freq")
            .assign(pct=lambda x: x.freq / len(df) * 100)
        )
        _save_table(counts, cat_counts_dir / f"{col}_counts")

    # 7. Target distribution plot
    print("\nPlotting target distribution...")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="damage_grade", data=df, order=[1, 2, 3], ax=ax)
    ax.set_title("Target distribution (damage_grade)")
    _save_fig(fig, figs_dir / "target_distribution.png")

    # 8. Correlazioni & heatmap
    print("\nComputing correlations & heatmap... (can take ~30 s)")
    corr = df[num_cols + ["damage_grade"]].corr()
    _save_table(corr.round(3), tables_dir / "correlation_matrix")

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr.clip(-1, 1), cmap="coolwarm", vmax=0.7, vmin=-0.7, ax=ax2)
    ax2.set_title("Correlation heatmap (numeric features)")
    _save_fig(fig2, figs_dir / "correlation_heatmap.png")

    # 9. Salva lista feature
    print("\nWriting feature_lists.json...")
    feature_lists = {"cat_cols": cat_cols, "geo_cols": geo_cols, "num_cols": num_cols}
    with open(output_dir / "feature_lists.json", 'w', encoding='utf-8') as f:
        json.dump(feature_lists, f, indent=2)

    print(f"\nEDA completed. Results saved to {_human_path(output_dir)}\n")


###############################################################################
# CLI entry-point
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA for the Richter's Predictor dataset."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/eda"),
        help="Directory to save tables & figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)
    run_eda(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()