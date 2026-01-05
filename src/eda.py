"""Exploratory data analysis utilities.

Generates histograms for numeric features, a correlation heatmap, and class balance plot.
Saves PNGs to the output directory.

Usage:
    python src/eda.py --dataset DataSet --target num --out outputs/eda
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _get_feature_names_from_column_transformer(ct, input_features=None):
    """Robustly extract feature names from a fitted ColumnTransformer.

    Tries get_feature_names_out first, then falls back to walking transformers_.
    """
    # sklearn >=1.0
    try:
        return ct.get_feature_names_out(input_features)
    except Exception:
        pass

    feature_names = []
    # ct.transformers_ is a list of (name, transformer, columns)
    if not hasattr(ct, "transformers_"):
        return feature_names

    for name, trans, cols in ct.transformers_:
        if trans == "drop":
            continue
        if trans == "passthrough":
            # passthrough columns: cols may be list of names
            if cols == "remainder":
                continue
            try:
                for col in cols:
                    feature_names.append(col)
            except Exception:
                continue
            continue

        transformer = trans
        # if pipeline, get the last step
        try:
            if hasattr(transformer, "named_steps"):
                last = list(transformer.named_steps.values())[-1]
            else:
                last = transformer
        except Exception:
            last = transformer

        # OneHotEncoder and similar have categories_ attribute
        if hasattr(last, "get_feature_names_out"):
            try:
                # prefer calling with original column names when supported
                names = last.get_feature_names_out(cols)
            except Exception:
                try:
                    names = last.get_feature_names_out()
                except Exception:
                    names = None
            if names is not None:
                feature_names.extend(list(names))
                continue

        # fallback: try to expand categories (for OneHotEncoder older versions)
        try:
            if hasattr(last, "categories_"):
                for col, cats in zip(cols, last.categories_):
                    for cat in cats:
                        feature_names.append(f"{col}_{cat}")
                continue
        except Exception:
            pass

        # last resort: add the column names themselves
        try:
            for col in cols:
                feature_names.append(col)
        except Exception:
            continue

    return feature_names


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="DataSet")
    p.add_argument("--target", type=str, default=None, help="Target column name or let loader auto-detect by omitting")
    p.add_argument("--model", type=str, default=None, help="Optional trained model (joblib) to compute feature importances")
    p.add_argument("--html", action="store_true", help="Also generate a simple HTML report that embeds generated images")
    p.add_argument("--out", type=str, default="outputs/eda")
    return p.parse_args(argv)


def load_data(dataset_dir: str, target_col: str = None):
    # reuse data loader
    # ensure project root importable when running script directly
    from pathlib import Path
    import sys
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.data import load_dataset

    X, y = load_dataset(dataset_dir, target_col=target_col)
    df = X.copy()
    df["__target__"] = y
    return df


def run(args=None):
    if args is None:
        args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.dataset, target_col=args.target)

    # numeric histograms
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "__target__" in num_cols:
        num_cols.remove("__target__")

    for c in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram: {c}")
        fig.savefig(out_dir / f"hist_{c}.png")
        plt.close(fig)

    # correlation heatmap (numeric)
    if len(num_cols) >= 2:
        corr = df[num_cols + ["__target__"]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation matrix")
        fig.savefig(out_dir / "correlation_heatmap.png")
        plt.close(fig)

    # class balance
    fig, ax = plt.subplots()
    sns.countplot(x="__target__", data=df, ax=ax)
    ax.set_title("Class balance")
    fig.savefig(out_dir / "class_balance.png")
    plt.close(fig)

    # Pairplot (sample if too many numeric columns)
    try:
        if len(num_cols) >= 2:
            pp_cols = num_cols
            if len(pp_cols) > 8:
                # pick top-variance numeric columns to keep the plot readable
                variances = df[pp_cols].var().sort_values(ascending=False)
                pp_cols = variances.index[:8].tolist()
            pairplot_path = out_dir / "pairplot.png"
            sns_pair = sns.pairplot(df[pp_cols + ["__target__"]], hue="__target__", corner=True, kind="scatter")
            sns_pair.fig.suptitle("Pairplot (sampled features)", y=1.02)
            sns_pair.savefig(pairplot_path)
            plt.close("all")
    except Exception:
        pass

    # Per-class distributions for numeric columns
    try:
        classes = sorted(df["__target__"].dropna().unique())
        for c in num_cols:
            fig, ax = plt.subplots()
            for cls in classes:
                sns.kdeplot(df.loc[df["__target__"] == cls, c].dropna(), label=str(cls), ax=ax, fill=False)
            ax.set_title(f"Per-class distribution: {c}")
            ax.legend(title="class")
            fig.savefig(out_dir / f"perclass_{c}.png")
            plt.close(fig)
    except Exception:
        pass

    # Feature importances if model provided
    if args.model:
        try:
            import joblib

            model = joblib.load(args.model)
            # Try to extract feature names from the preprocessor
            feature_names = None
            try:
                pre = model.named_steps.get("pre") if hasattr(model, "named_steps") else None
                if pre is not None:
                    try:
                        feature_names = _get_feature_names_from_column_transformer(pre)
                    except Exception:
                        feature_names = None
            except Exception:
                feature_names = None

            # If classifier has feature_importances_
            clf = None
            try:
                clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
            except Exception:
                clf = None

            if clf is not None and hasattr(clf, "feature_importances_"):
                import numpy as _np

                importances = _np.array(clf.feature_importances_)
                if feature_names is None or len(feature_names) != len(importances):
                    # fallback to simple indices
                    feature_names = [f"f{i}" for i in range(len(importances))]

                fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                topk = fi[:min(50, len(fi))]
                names, vals = zip(*topk)
                fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.2)))
                ax.barh(list(reversed(names)), list(reversed(vals)))
                ax.set_title("Feature importances (top features)")
                fig.tight_layout()
                fig.savefig(out_dir / "feature_importances.png")
                plt.close(fig)
        except Exception:
            pass

    # Simple HTML report bundling images
    if args.html:
        try:
            imgs = [p for p in sorted(out_dir.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            html_lines = ["<html>", "<head><meta charset=\"utf-8\"><title>EDA Report</title></head>", "<body>", f"<h1>EDA Report for {args.dataset}</h1>"]
            for im in imgs:
                html_lines.append(f"<h3>{im.name}</h3>")
                html_lines.append(f"<img src=\"{im.name}\" style=\"max-width:100%;height:auto\">")
            html_lines.append("</body></html>")
            with open(out_dir / "report.html", "w") as fh:
                fh.write("\n".join(html_lines))
        except Exception:
            pass

    print(f"EDA artifacts written to {out_dir}")


if __name__ == "__main__":
    run()
