"""Exploratory data analysis utilities.

Generates histograms for numeric features, a correlation heatmap, and class balance plot.
Saves PNGs to the output directory.

"""

import sys
import os

import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from src.dataRead import load_dataset
from src.dataClean import clean_data

output_dir = "../outputs/eda"

def dataEDA(data, model_path=None, generate_html=True):
    """
    Comprehensive EDA: Distributions, Correlations, Class Balance, 
    Pairplots, PCA, Feature Importance, and HTML Reporting.
    """
    df = data.copy()

    out_dir = Path(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sns.set_theme(style="whitegrid")
    target_col = 'target' # Standardized target name
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # 1. Class Balance
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_col, data=df, palette="viridis")
    plt.title("Class Balance")
    plt.savefig(out_dir / "0_class_balance.png")
    plt.close()

    # 2. Individual Histograms (KDE included)
    for c in num_cols:
        plt.figure()
        sns.histplot(df[c].dropna(), kde=True, color='teal')
        plt.title(f"Histogram: {c}")
        plt.savefig(out_dir / f"hist_{c}.png")
        plt.close()

    # 3. Correlation Heatmap
    if len(num_cols) >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[num_cols + [target_col]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.savefig(out_dir / '1_heatmap.png')
        plt.close()

    # 4. Pairplot (with Variance-based Filtering)
    if len(num_cols) >= 2:
        pp_cols = num_cols
        if len(pp_cols) > 8:
            variances = df[pp_cols].var().sort_values(ascending=False)
            pp_cols = variances.index[:8].tolist()
        
        sns_pair = sns.pairplot(df[pp_cols + [target_col]], hue=target_col, corner=True, palette='bright')
        sns_pair.fig.suptitle("Pairplot (Top Variance Features)", y=1.02)
        sns_pair.savefig(out_dir / "2_pairplot.png")
        plt.close("all")

    # 5. Per-Class KDE Distributions
    classes = sorted(df[target_col].dropna().unique())
    for c in num_cols:
        plt.figure()
        for cls in classes:
            sns.kdeplot(df.loc[df[target_col] == cls, c].dropna(), label=f"Class {cls}", fill=True, alpha=0.3)
        plt.title(f"Per-class distribution: {c}")
        plt.legend()
        plt.savefig(out_dir / f"perclass_{c}.png")
        plt.close()

    # 6. PCA (Dimensionality Reduction)
    try:
        pca_df = df.dropna()
        features = pca_df[num_cols]
        scaled = StandardScaler().fit_transform(features)
        pca_data = PCA(n_components=2).fit_transform(scaled)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=pca_df[target_col], palette='viridis')
        plt.title('PCA Projection')
        plt.savefig(out_dir / '3_pca_clusters.png')
        plt.close()
    except Exception as e:
        print(f"PCA failed: {e}")

    # 7. Feature Importance (If Model Provided)
    if model_path:
        try:
            import joblib
            model = joblib.load(model_path)
            # Logic to extract feature importances from a sklearn-style model
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.figure(figsize=(10, 6))
                plt.title("Feature Importances")
                plt.bar(range(len(importances)), importances[indices], align="center")
                plt.xticks(range(len(importances)), [num_cols[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig(out_dir / "4_feature_importances.png")
                plt.close()
        except Exception as e:
            print(f"Feature importance failed: {e}")

    # 8. HTML Report Generation
    if generate_html:
        imgs = [p for p in sorted(out_dir.iterdir()) if p.suffix.lower() in (".png", ".jpg")]
        html_content = [f"<html><head><title>EDA Report</title></head><body><h1>EDA Visual Report</h1>"]
        for im in imgs:
            html_content.append(f"<h3>{im.name}</h3><img src='{im.name}' style='max-width:800px;'><br>")
        html_content.append("</body></html>")
        with open(out_dir / "report.html", "w") as f:
            f.write("\n".join(html_content))

    print(f"✅ EDA complete. Results saved in: {out_dir}")


if __name__ == "__main__":
    data = load_dataset()

    data = clean_data(data)
    dataEDA(data, model_path=None, generate_html=True)
