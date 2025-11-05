#!/usr/bin/env python3
"""
Explorative EDA + Cleaning + Spatial Imputation (Haversine) + Feature Engineering
--------------------------------------------------------------------------------
Auto-run version (no CLI). It will:

1) Load CSV from DATA_PATH.
2) EDA: overview, missingness, distributions, correlation heatmap, pairplot, outlier counts.
3) Spatially impute all `lashade_*` columns using geographic nearest neighbors (BallTree, haversine).
4) Drop only non-lashade columns with > MISSING_THRESH missingness.
5) Impute remaining NaNs (numeric median, categorical mode).
6) Engineer features:
   - canopy_gap
   - canopy_percent_of_goal
   - env_heat_composite
   - avg_transport_access
7) Prune highly collinear numeric columns (|r| > 0.95), while protecting engineered features.
8) Save cleaned dataset, plots, and logs to OUTPUT_DIR.

Dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree

# =========================
# CONFIG (EDIT THESE PATHS)
# =========================
DATA_PATH = "/Users/jy/Documents/USC/csci461/la-shade-optimization/shade_optimization_data.csv"
OUTPUT_DIR = Path("/Users/jy/Documents/USC/csci461/la-shade-optimization/eda_outputs")
MISSING_THRESH = 0.60     # drop non-lashade columns if >60% missing
K_NEIGHBORS = 1           # 1 = strict nearest; use 3–5 for local average fill

# -------------- helpers: visuals --------------
def save_missingness_bar(df: pd.DataFrame, outdir: Path, topn: int = 25, name_suffix=""):
    miss_frac = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=miss_frac.head(topn).index, y=miss_frac.head(topn).values)
    plt.xticks(rotation=90)
    plt.ylabel("Missing fraction")
    plt.title(f"Top {topn} Columns by Missing Fraction{name_suffix}")
    plt.tight_layout()
    plt.savefig(outdir / f"missingness_top{topn}{name_suffix}.png")
    plt.close()
    return miss_frac

def plot_correlation_heatmap(df: pd.DataFrame, outdir: Path, name: str = "correlation_heatmap.png"):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        return
    corr = df[num_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(outdir / name)
    plt.close()

def plot_pairplot_key(df: pd.DataFrame, outdir: Path):
    key_vars = [v for v in [
        "urban_heat_idx", "urban_heat_idx_percentile",
        "tree_percent_w", "pm25", "pm25_percentile",
        "dist_to_metroline_1", "dist_to_busstop_1",
        "dist_to_vacant_park_1", "longitude", "latitude"
    ] if v in df.columns]
    if len(key_vars) >= 3:
        g = sns.pairplot(df[key_vars], diag_kind="kde", corner=True)
        g.fig.suptitle("Key Relationships (Heat, Canopy, Pollution, Transit)", y=1.02)
        g.savefig(outdir / "pairplot_key_relationships.png")
        plt.close("all")

def plot_distributions(df: pd.DataFrame, outdir: Path, cols: list[str]):
    for col in cols:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            plt.figure(figsize=(5, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(outdir / f"hist_{col}.png")
            plt.close()

            plt.figure(figsize=(5, 4))
            plt.boxplot(df[col].dropna(), vert=True)
            plt.ylabel(col)
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.savefig(outdir / f"box_{col}.png")
            plt.close()

def zscore_counts_bar(df: pd.DataFrame, outdir: Path, name="outliers_zscore.png"):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return
    z = (num - num.mean()) / num.std(ddof=0)
    out_counts = (np.abs(z) > 3).sum()
    out_counts = out_counts[out_counts > 0]
    if not out_counts.empty:
        plt.figure(figsize=(10, 4))
        sns.barplot(x=out_counts.index, y=out_counts.values)
        plt.xticks(rotation=90)
        plt.title("Outlier Counts (|Z| > 3)")
        plt.tight_layout()
        plt.savefig(outdir / name)
        plt.close()

# -------------- helpers: cleaning/imputation --------------
def spatial_impute_haversine(df: pd.DataFrame, lashade_cols: list[str],
                             lat_col="latitude", lon_col="longitude", k=1):
    """
    For each lashade_* column:
      - Build a BallTree on KNOWN rows (lat/lon radians)
      - For MISSING rows, query nearest k and fill with nearest value (k=1) or mean(k)
    """
    assert lat_col in df.columns and lon_col in df.columns, "lat/lon columns are required for spatial imputation."
    coords_all = np.radians(df[[lat_col, lon_col]].to_numpy())

    for col in lashade_cols:
        miss_mask = df[col].isna()
        if not miss_mask.any():
            continue
        known_mask = df[col].notna()
        if not known_mask.any():
            continue

        known_coords = coords_all[known_mask.values]
        tree = BallTree(known_coords, metric="haversine")

        missing_coords = coords_all[miss_mask.values]
        dist, ind = tree.query(missing_coords, k=min(k, known_coords.shape[0]))
        if k == 1 or ind.ndim == 1:
            filled_vals = df.loc[known_mask, col].to_numpy()[ind.flatten()]
        else:
            neighbor_vals = df.loc[known_mask, col].to_numpy()[ind]
            filled_vals = np.nanmean(neighbor_vals, axis=1)

        df.loc[miss_mask, col] = filled_vals

    return df

def drop_high_missing_except_lashade(df: pd.DataFrame, thresh: float):
    miss_frac = df.isna().mean()
    lashade_cols = [c for c in df.columns if c.startswith("lashade_")]
    drop_candidates = miss_frac[(miss_frac > thresh) & (~miss_frac.index.isin(lashade_cols))].index.tolist()
    return drop_candidates

# =========================
# MAIN (auto-run; no CLI)
# =========================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # ---- Overview ----
    print("\n=== DATA OVERVIEW ===")
    print(df.info())
    print("\n=== SUMMARY STATS (numeric head) ===")
    print(df.select_dtypes(include=[np.number]).describe().T.head(15))

    # ---- Missingness (before) ----
    save_missingness_bar(df, OUTPUT_DIR, name_suffix="_before_imputation")

    # ---- Spatial imputation for LA Shade ----
    lashade_cols = [c for c in df.columns if c.startswith("lashade_")]
    if lashade_cols and {"latitude", "longitude"}.issubset(df.columns):
        print(f"\n🌳 Spatially imputing {len(lashade_cols)} LA Shade columns using haversine k={K_NEIGHBORS} ...")
        df = spatial_impute_haversine(df, lashade_cols, "latitude", "longitude", k=K_NEIGHBORS)
    else:
        print("\n(No LA Shade columns or missing lat/lon — skipping spatial imputation.)")

    # ---- Missingness (after spatial for lashade) ----
    after_spatial_miss = df.isna().mean().sort_values(ascending=False)
    print("\nTop missingness AFTER spatial impute (first 15 cols):")
    print(after_spatial_miss.head(15))

    # ---- Drop high-missing non-lashade columns ----
    drop_cols = drop_high_missing_except_lashade(df, MISSING_THRESH)
    if drop_cols:
        print(f"\n📉 Dropping {len(drop_cols)} non-LA Shade columns (> {MISSING_THRESH*100:.1f}% missing):")
        print(drop_cols)
        df.drop(columns=drop_cols, inplace=True)
    else:
        print(f"\nNo non-LA Shade columns exceeded missingness threshold {MISSING_THRESH:.2f}.")

    # ---- Handle remaining NaNs ----
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    # ---- EDA Visuals ----
    key_cols = [c for c in [
        "urban_heat_idx", "urban_heat_idx_percentile",
        "tree_percent_w", "pm25", "pm25_percentile",
        "dist_to_metroline_1", "dist_to_busstop_1",
        "dist_to_vacant_park_1", "longitude", "latitude"
    ] if c in df.columns]

    plot_distributions(df, OUTPUT_DIR, key_cols)
    plot_correlation_heatmap(df, OUTPUT_DIR, name="correlation_heatmap_after_imputation.png")
    plot_pairplot_key(df, OUTPUT_DIR)
    zscore_counts_bar(df, OUTPUT_DIR)

    # ---- Engineered features (always added when sources present) ----
    engineered = []

    if {"lashade_tc_goal", "lashade_treecanopy"}.issubset(df.columns):
        df["canopy_gap"] = df["lashade_tc_goal"] - df["lashade_treecanopy"]
        df["canopy_percent_of_goal"] = df["lashade_treecanopy"] / (df["lashade_tc_goal"] + 1e-6)
        engineered += ["canopy_gap", "canopy_percent_of_goal"]

    if {"urban_heat_idx", "pm25", "tree_percent_w"}.issubset(df.columns):
        df["env_heat_composite"] = (
            df["urban_heat_idx"].rank(pct=True) * 0.5
            + df["pm25"].rank(pct=True) * 0.3
            - df["tree_percent_w"].rank(pct=True) * 0.2
        )
        engineered.append("env_heat_composite")

    if {"dist_to_busstop_1", "dist_to_metrostop_1", "dist_to_vacant_park_1"}.issubset(df.columns):
        df["avg_transport_access"] = df[[
            "dist_to_busstop_1", "dist_to_metrostop_1", "dist_to_vacant_park_1"
        ]].mean(axis=1)
        engineered.append("avg_transport_access")

    if engineered:
        print(f"\n✅ Added engineered features: {engineered}")
        with open(OUTPUT_DIR / "engineered_features.txt", "w") as f:
            for c in engineered:
                f.write(c + "\n")
    else:
        print("\n⚠️ No engineered features were added (missing required source columns).")

    # ---- Prune highly collinear numeric features (|r| > 0.95), but protect engineered ----
    protect_cols = set(["canopy_gap", "canopy_percent_of_goal", "env_heat_composite", "avg_transport_access"])
    num = df.select_dtypes(include=[np.number])
    to_drop_corr = []
    if not num.empty:
        corr_abs = num.corr().abs()
        upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
        # Only consider columns NOT in protect_cols
        candidates = [c for c in upper.columns if c not in protect_cols]
        for c in candidates:
            # also avoid dropping any protected row labels
            mask_series = upper[c].drop(labels=list(protect_cols & set(upper.index)), errors="ignore")
            if (mask_series > 0.95).any():
                to_drop_corr.append(c)

    if to_drop_corr:
        print(f"\n📉 Dropping {len(to_drop_corr)} highly correlated columns (>0.95):")
        print(to_drop_corr)
        df.drop(columns=to_drop_corr, inplace=True)

    # ---- Prove engineered features exist + small preview ----
    print("\n🔎 Engineered features now in dataframe:")
    for feat in engineered:
        if feat in df.columns:
            print(f"   - {feat} (non-null count: {df[feat].notna().sum()})")

    preview_cols = ["latitude","longitude"] + [c for c in engineered if c in df.columns]
    if preview_cols:
        df[preview_cols].head(20).to_csv(OUTPUT_DIR / "engineered_preview.csv", index=False)

    # ---- Save cleaned outputs ----
    out_clean = OUTPUT_DIR / "data_cleaned.csv"
    df.to_csv(out_clean, index=False)

    with open(OUTPUT_DIR / "final_feature_list.txt", "w") as f:
        for c in df.columns:
            f.write(c + "\n")

    dropped_log = OUTPUT_DIR / "dropped_columns.txt"
    with open(dropped_log, "w") as f:
        if to_drop_corr:
            f.write("Dropped due to high correlation:\n")
            for c in to_drop_corr:
                f.write(f"- {c}\n")
        # We already printed missingness-dropped cols above; optionally record them too:
        # (Recompute last drop list if desired)
        # ...

    print("\n--- DONE ---")
    print(f"💾 Cleaned dataset: {out_clean}")
    print(f"🖼️ Visualizations saved in: {OUTPUT_DIR.resolve()}")
    print(f"📝 Logs saved: engineered_features.txt, final_feature_list.txt, dropped_columns.txt")

if __name__ == "__main__":
    main()
