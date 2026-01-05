"""CLI for training the model."""
import argparse
import json
import os
import sys
from pathlib import Path

# When running as a script (python src/train.py) ensure project root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import load_dataset
from src.model import train_and_evaluate, save_model
import sklearn
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a scikit-learn model on CSV data in a folder")
    p.add_argument("--dataset", type=str, default="DataSet", help="Path to dataset folder containing CSV files")
    p.add_argument("--target", type=str, default=None, help="Name of the target column in CSV files. If omitted the loader will attempt to auto-detect (use 'auto' to force detection).")
    p.add_argument("--output", type=str, default="models/model.joblib", help="Path to save trained model")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=100, help="Number of trees for RandomForest")
    p.add_argument("--max-depth", type=int, default=None, help="Max depth for RandomForest (None for unlimited)")
    p.add_argument("--search", type=str, default="none", choices=["none", "grid", "random"], help="Hyperparameter search type: none, grid, or random")
    p.add_argument("--param-grid-file", type=str, default=None, help="Optional JSON file containing param grid for search")
    p.add_argument("--search-cv", type=int, default=3, help="CV folds for hyperparameter search")
    p.add_argument("--n-iter", type=int, default=10, help="Number of parameter settings sampled for RandomizedSearchCV")
    p.add_argument("--use-halving", action="store_true", help="Use successive halving search (HalvingRandomSearchCV) for randomized search")
    p.add_argument("--progress", action="store_true", help="Show progress bar and enable early stopping for randomized search (manual search)")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience for manual randomized search (iterations with no improvement)")
    p.add_argument("--mlflow-log-cv", action="store_true", default=True, help="Log full cv_results.csv to MLflow when search is used (default: True)")
    p.add_argument("--mlflow-child-runs", action="store_true", default=False, help="Create MLflow nested child runs for each candidate in the CV results")
    p.add_argument("--classifier", type=str, default="rf", choices=["rf", "logreg", "svm"], help="Classifier to use: rf (RandomForest), logreg (LogisticRegression), svm (SVC)")
    p.add_argument("--C", type=float, default=1.0, help="Regularization strength for LogisticRegression or SVC")
    p.add_argument("--kernel", type=str, default="rbf", help="Kernel for SVC (if classifier=svm)")
    p.add_argument("--compare-models", action="store_true", help="Train and evaluate both LogisticRegression and RandomForest and produce a comparison report")
    p.add_argument("--export-features", action="store_true", help="Export the final transformed feature matrix (after preprocessor) as CSV next to the model")
    return p.parse_args(argv)


def run(args=None):
    if args is None:
        args = parse_args()
    # Load data
    X, y = load_dataset(args.dataset, target_col=args.target)

    # Train (pass classifier with chosen hyperparameters)
    # Build classifier according to CLI selection
    clf = None
    if args.classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)
    elif args.classifier == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=args.C, max_iter=1000, random_state=args.random_state)
    elif args.classifier == "svm":
        from sklearn.svm import SVC

        clf = SVC(C=args.C, kernel=args.kernel, probability=True, random_state=args.random_state)
    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")
    # If hyperparameter search requested, run search (GridSearchCV or RandomizedSearchCV) over a pipeline
    from src.model import build_pipeline
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = None, None, None, None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    numeric_cols = list(X_train.select_dtypes(include=["number"]).columns)
    categorical_cols = list(X_train.select_dtypes(include=["object", "category"]).columns)

    if args.search != "none":
        # build base pipeline
        base_pipe = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols, classifier=clf)
        # load param grid
        if args.param_grid_file:
            try:
                with open(args.param_grid_file, "r") as fh:
                    param_grid = json.load(fh)
            except Exception:
                param_grid = None
        else:
            param_grid = None

        if param_grid is None:
            # default grid only for RandomForest classifier (use clf__ prefix to reference estimator inside pipeline)
            if args.classifier == "rf":
                param_grid = {
                    "clf__n_estimators": [50, 100, 200],
                    "clf__max_depth": [None, 5, 10],
                }
            elif args.classifier == "logreg":
                param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
            elif args.classifier == "svm":
                param_grid = {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]}

        # choose searcher
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        searcher = None
        cv_path = None
        best_params_path = None
        # If user requested a manual progress-enabled random search
        if args.search == "random" and args.progress:
            # manual randomized search: sample settings, cross-validate, allow early stopping
            try:
                from sklearn.base import clone
                from sklearn.model_selection import cross_val_score
                import random as _random
                try:
                    from tqdm.auto import tqdm
                except Exception:
                    def tqdm(x, **kw):
                        return x

                seen = set()
                results = []
                best_score = -1e9
                no_improve = 0
                n_samples = args.n_iter
                for i in tqdm(range(n_samples), desc="Random search"):
                    # sample params
                    params = {}
                    for k, vals in param_grid.items():
                        # if list-like sample uniformly
                        if isinstance(vals, list):
                            v = _random.choice(vals)
                        else:
                            # fallback: use as-is
                            v = vals
                        params[k] = v
                    tup = tuple(sorted(params.items()))
                    if tup in seen:
                        continue
                    seen.add(tup)
                    estimator = clone(base_pipe)
                    try:
                        estimator.set_params(**params)
                    except Exception:
                        pass
                    # cross-validate
                    try:
                        scores = cross_val_score(estimator, X_train, y_train, cv=args.search_cv, n_jobs=1)
                        mean_score = float(scores.mean())
                        std_score = float(scores.std())
                    except Exception:
                        mean_score = float("nan")
                        std_score = float("nan")

                    results.append({"params": params, "mean_test_score": mean_score, "std_test_score": std_score})
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                        no_improve = 0
                    else:
                        no_improve += 1
                    if args.patience and no_improve >= args.patience:
                        break

                # refit best model on full train
                best_model = clone(base_pipe)
                try:
                    best_model.set_params(**best_params)
                except Exception:
                    pass
                best_model.fit(X_train, y_train)

                # save results
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_model(best_model, str(out_path))
                import pandas as _pd

                cv_df = _pd.DataFrame([{**r["params"], "mean_test_score": r["mean_test_score"], "std_test_score": r["std_test_score"]} for r in results])
                cv_path = out_path.parent / "cv_results.csv"
                cv_df.to_csv(cv_path, index=False)

                best_params_path = out_path.parent / "best_params.json"
                with open(best_params_path, "w") as fh:
                    json.dump(best_params, fh, indent=2)

                metrics = {
                    "accuracy": float(np.nan),
                    "report": "Search completed (manual); evaluate on test set saved separately.",
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "search": args.search,
                    "best_score_cv": float(best_score) if best_score is not None else None,
                }
                model = best_model
                metrics_path = out_path.parent / "metrics.json"
                with open(metrics_path, "w") as fh:
                    json.dump(metrics, fh, indent=2)
                print(f"Saved best model to {out_path}")
                print(f"Saved cv results to {cv_path}")
            except Exception as e:
                print("Manual randomized search failed:", e)
                raise
        else:
            # use sklearn searchers (Grid or Randomized or HalvingRandomized)
            if args.search == "grid":
                searcher = GridSearchCV(base_pipe, param_grid, cv=args.search_cv, n_jobs=1, verbose=2, refit=True)
            else:
                if args.use_halving:
                    try:
                        from sklearn.model_selection import HalvingRandomSearchCV

                        searcher = HalvingRandomSearchCV(base_pipe, param_grid, cv=args.search_cv, factor=2, verbose=2, random_state=args.random_state)
                    except Exception:
                        searcher = RandomizedSearchCV(base_pipe, param_grid, cv=args.search_cv, n_iter=args.n_iter, n_jobs=1, verbose=2, refit=True, random_state=args.random_state)
                else:
                    searcher = RandomizedSearchCV(base_pipe, param_grid, cv=args.search_cv, n_iter=args.n_iter, n_jobs=1, verbose=2, refit=True, random_state=args.random_state)

            print("Running hyperparameter search... this may take a while")
            searcher.fit(X_train, y_train)

            best_model = searcher.best_estimator_
            # evaluate on held-out test
            preds = best_model.predict(X_test)
            from sklearn.metrics import accuracy_score, classification_report

            acc = float(accuracy_score(y_test, preds))
            try:
                report = classification_report(y_test, preds, zero_division=0)
            except Exception:
                report = "Classification report not available"

            metrics = {
                "accuracy": acc,
                "report": report,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "search": args.search,
                "best_score_cv": float(searcher.best_score_) if hasattr(searcher, "best_score_") else None,
            }

            # Save best model
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_model(best_model, str(out_path))
            # save search CV results
            try:
                import pandas as _pd

                cv_df = _pd.DataFrame(searcher.cv_results_)
                cv_path = out_path.parent / "cv_results.csv"
                cv_df.to_csv(cv_path, index=False)
            except Exception:
                cv_path = None

            # save best params
            best_params_path = out_path.parent / "best_params.json"
            with open(best_params_path, "w") as fh:
                json.dump(searcher.best_params_, fh, indent=2)

            print(f"Saved best model to {out_path}")
            print(f"Search best score (cv): {searcher.best_score_}")
            # set model variable for downstream MLflow logging to reference
            model = best_model
            metrics_path = out_path.parent / "metrics.json"
            with open(metrics_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
            print(f"Saved metrics to {metrics_path}")

    else:
        model, metrics = train_and_evaluate(X, y, test_size=args.test_size, random_state=args.random_state, classifier=clf)
        # Ensure output dir exists
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, str(out_path))

        # Print metrics and save metrics.json next to model
        print(f"Saved model to {out_path}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(metrics["report"])

        metrics_path = out_path.parent / "metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Saved metrics to {metrics_path}")

    # At this point `model`, `metrics`, `out_path`, `cv_path`, and `best_params_path` may be set
    # Ensure output dir exists and model saved (for branches that didn't save earlier)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_model(model, str(out_path))
    except Exception:
        pass

    # Print metrics and save metrics.json next to model (if not already saved)
    print(f"Saved model to {out_path}")
    try:
        print(f"Accuracy: {metrics.get('accuracy', float('nan')):.4f}")
    except Exception:
        pass
    if "report" in metrics:
        try:
            print(metrics["report"])
        except Exception:
            pass

    metrics_path = out_path.parent / "metrics.json"
    try:
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Saved metrics to {metrics_path}")
    except Exception:
        pass

    # Save model metadata for reproducibility
    meta = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "model_params": {},
        "random_state": args.random_state,
    }
    try:
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            meta["model_params"] = model.named_steps["clf"].get_params()
    except Exception:
        meta["model_params"] = {}
    # include git commit hash if available
    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None

    meta_path = out_path.parent / "model_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved model metadata to {meta_path}")

    # If compare-models requested: evaluate LogisticRegression and RandomForest with cross-validation
    if args.compare_models:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_validate

            # Build pipelines for both classifiers using detected columns
            pre_numeric_cols = numeric_cols
            pre_categorical_cols = categorical_cols

            lr_clf = LogisticRegression(C=args.C, max_iter=1000, random_state=args.random_state)
            rf_clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)

            from src.model import build_pipeline
            pipe_lr = build_pipeline(pre_numeric_cols, pre_categorical_cols, classifier=lr_clf)
            pipe_rf = build_pipeline(pre_numeric_cols, pre_categorical_cols, classifier=rf_clf)

            # Determine scoring metrics: include roc_auc (use ovR for multiclass)
            unique_classes = sorted(list(set(y)))
            if len(unique_classes) == 2:
                roc_scoring = "roc_auc"
            else:
                roc_scoring = "roc_auc_ovr"

            scoring = {"accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro", "roc_auc": roc_scoring}

            print("Running cross-validation for model comparison...")
            cv_lr = cross_validate(pipe_lr, X, y, cv=args.search_cv, scoring=scoring, return_train_score=False)
            cv_rf = cross_validate(pipe_rf, X, y, cv=args.search_cv, scoring=scoring, return_train_score=False)

            def summarize_cv(cv_res):
                out = {}
                for k, v in cv_res.items():
                    if k.startswith("test_"):
                        metric = k.replace("test_", "")
                        try:
                            out[metric] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        except Exception:
                            out[metric] = {"mean": None, "std": None}
                return out

            summary = {"LogisticRegression": summarize_cv(cv_lr), "RandomForest": summarize_cv(cv_rf)}

            comp_path = out_path.parent / "model_comparison.json"
            with open(comp_path, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"Saved model comparison to {comp_path}")

            # Write short markdown documenting model selection & tuning
            try:
                md_lines = ["# Model selection report", "", "Trained and compared two classifiers: Logistic Regression and Random Forest.", "", "## Cross-validation results (mean ± std)", ""]
                for mname, stats in summary.items():
                    md_lines.append(f"### {mname}")
                    for metric, vals in stats.items():
                        md_lines.append(f"- {metric}: {vals.get('mean'):.4f} ± {vals.get('std'):.4f}")
                    md_lines.append("")
                md_lines.append("## Tuning process")
                md_lines.append("The default parameter grids were used when performing hyperparameter search (if enabled). For Logistic Regression we tuned regularization C; for Random Forest we tuned n_estimators and max_depth. Cross-validation used scikit-learn's cross_validate with the scoring metrics described above.")
                md_path = out_path.parent / "model_selection.md"
                with open(md_path, "w") as fh:
                    fh.write("\n".join(md_lines))
                print(f"Saved model selection report to {md_path}")
            except Exception:
                pass

        except Exception as e:
            print("Model comparison failed:", e)

    # Optionally export transformed features (fit preprocessor on full data and dump transformed X)
    if args.export_features:
        try:
            # Build a pipeline (without training classifier) to get preprocessor
            from src.model import build_pipeline
            dummy_clf = RandomForestClassifier(n_estimators=1)
            final_pipe = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols, classifier=dummy_clf)
            # fit only preprocessor by calling fit on the pipeline and then using named_steps['pre']
            final_pipe.fit(X, y)
            pre = final_pipe.named_steps.get("pre")
            transformed = pre.transform(X)
            # convert to dense if necessary
            transformed = np.array(transformed.todense()) if hasattr(transformed, "todense") else np.asarray(transformed)
            # get feature names
            try:
                from src.eda import _get_feature_names_from_column_transformer
                feat_names = _get_feature_names_from_column_transformer(pre)
            except Exception:
                feat_names = [f"f{i}" for i in range(transformed.shape[1])]

            import pandas as _pd
            df_feat = _pd.DataFrame(transformed, columns=feat_names)
            feats_path = out_path.parent / "transformed_features.csv"
            df_feat.to_csv(feats_path, index=False)
            print(f"Saved transformed features to {feats_path}")
        except Exception as e:
            print("Exporting transformed features failed:", e)

    # Log to MLflow (parameters, metrics, artifacts) — optional via ENABLE_MLFLOW env var
    ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "1").lower() in ("1", "true", "yes")
    if ENABLE_MLFLOW:
        try:
            # Ensure MLflow tracking URI is set to local mlruns by default when not provided
            if not os.getenv("MLFLOW_TRACKING_URI"):
                local_store = Path.cwd() / "mlruns"
                local_uri = f"file:{local_store}"
                mlflow.set_tracking_uri(local_uri)
                print(f"MLflow tracking URI not set; using local: {local_uri}")
            else:
                print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")

            exp_name = os.getenv("MLFLOW_EXPERIMENT", "default")
            exp = mlflow.set_experiment(exp_name)
            print(f"Using MLflow experiment: id={exp.experiment_id}, name={exp.name}")

            run_name = f"{Path(args.dataset).name}-{args.classifier}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
            with mlflow.start_run(run_name=run_name):
                # set tags
                mlflow.set_tag("dataset", Path(args.dataset).name)
                try:
                    import subprocess

                    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
                    mlflow.set_tag("git_branch", branch)
                except Exception:
                    pass

                # params
                mlflow.log_param("n_estimators", args.n_estimators)
                mlflow.log_param("max_depth", args.max_depth)
                mlflow.log_param("test_size", args.test_size)
                mlflow.log_param("random_state", args.random_state)
                mlflow.log_param("classifier", args.classifier)
                mlflow.log_param("C", args.C)
                mlflow.log_param("kernel", args.kernel)
                # model params (flattened)
                try:
                    for k, v in meta.get("model_params", {}).items():
                        try:
                            mlflow.log_param(f"model_{k}", str(v))
                        except Exception:
                            pass
                except Exception:
                    pass

                # metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))

                # log artifacts: metrics.json and model_metadata.json
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(meta_path))

                # log model via mlflow.sklearn (this will also save the model artifact)
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception:
                    mlflow.log_artifact(str(out_path))

                # If a search was used, log cv results and best params and optionally create child runs
                try:
                    if args.search != "none":
                        # log cv_results.csv and best_params.json if they exist
                        try:
                            if cv_path is not None:
                                # log cv results to MLflow by default
                                mlflow.log_artifact(str(cv_path))
                        except Exception:
                            pass
                        try:
                            if best_params_path is not None:
                                mlflow.log_artifact(str(best_params_path))
                        except Exception:
                            pass

                        # Optionally create nested runs per candidate
                        if args.mlflow_child_runs:
                            try:
                                # Prefer in-memory searcher.cv_results_ when available
                                if 'searcher' in locals() and searcher is not None and hasattr(searcher, 'cv_results_'):
                                    cv = searcher.cv_results_
                                    n = len(cv.get('params', []))
                                    for i in range(n):
                                        params_i = cv.get('params', [])[i]
                                        # metrics: mean_test_score, std_test_score, rank_test_score if available
                                        m = {}
                                        if 'mean_test_score' in cv:
                                            try:
                                                m['mean_test_score'] = float(cv['mean_test_score'][i])
                                            except Exception:
                                                pass
                                        if 'std_test_score' in cv:
                                            try:
                                                m['std_test_score'] = float(cv['std_test_score'][i])
                                            except Exception:
                                                pass
                                        if 'rank_test_score' in cv:
                                            try:
                                                m['rank_test_score'] = int(cv['rank_test_score'][i])
                                            except Exception:
                                                pass
                                        with mlflow.start_run(nested=True):
                                            # log params (flatten simple types)
                                            try:
                                                flat_params = {str(k): str(v) for k, v in params_i.items()}
                                                mlflow.log_params(flat_params)
                                            except Exception:
                                                pass
                                            for mk, mv in m.items():
                                                try:
                                                    mlflow.log_metric(mk, float(mv))
                                                except Exception:
                                                    pass
                                else:
                                    # Fallback: read cv_results.csv if available
                                    import pandas as _pd

                                    if cv_path is not None and cv_path.exists():
                                        df_cv = _pd.read_csv(cv_path)
                                        # assume parameter columns are all except mean_test_score/std_test_score
                                        for _, row in df_cv.iterrows():
                                            params_i = {k: row[k] for k in df_cv.columns if k not in ('mean_test_score', 'std_test_score')}
                                            m = {}
                                            if 'mean_test_score' in df_cv.columns:
                                                try:
                                                    m['mean_test_score'] = float(row['mean_test_score'])
                                                except Exception:
                                                    pass
                                            if 'std_test_score' in df_cv.columns:
                                                try:
                                                    m['std_test_score'] = float(row['std_test_score'])
                                                except Exception:
                                                    pass
                                            with mlflow.start_run(nested=True):
                                                try:
                                                    flat_params = {str(k): str(v) for k, v in params_i.items()}
                                                    mlflow.log_params(flat_params)
                                                except Exception:
                                                    pass
                                                for mk, mv in m.items():
                                                    try:
                                                        mlflow.log_metric(mk, float(mv))
                                                    except Exception:
                                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass

                # create a confusion matrix plot on the test set and log it
                try:
                    from sklearn.model_selection import train_test_split

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(set(y)) > 1 else None)
                    preds = model.predict(X_test)
                    cm = confusion_matrix(y_test, preds, labels=sorted(list(set(y_test))))
                    fig, ax = plt.subplots(figsize=(6, 4))
                    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    ax.set_ylabel("True label")
                    ax.set_xlabel("Predicted label")
                    ax.set_title("Confusion matrix")
                    labels = [str(l) for l in sorted(list(set(y_test)))]
                    ax.set_xticks(range(len(labels)))
                    ax.set_yticks(range(len(labels)))
                    ax.set_xticklabels(labels)
                    ax.set_yticklabels(labels)
                    plt.tight_layout()
                    plot_path = out_path.parent / "confusion_matrix.png"
                    fig.savefig(plot_path)
                    plt.close(fig)
                    mlflow.log_artifact(str(plot_path))
                except Exception:
                    pass

                # log ROC AUC and ROC plot when possible
                try:
                    from sklearn.preprocessing import label_binarize
                    from sklearn.metrics import roc_curve, auc, roc_auc_score

                    y_true = y_test
                    y_score = None
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(X_test)
                    elif hasattr(model, "decision_function"):
                        y_score = model.decision_function(X_test)

                    if y_score is not None:
                        classes = sorted(list(set(y_true)))
                        n_classes = len(classes)
                        if n_classes == 2:
                            try:
                                pos_idx = list(model.classes_).index(classes[1]) if hasattr(model, "classes_") else 1
                            except Exception:
                                pos_idx = 1
                            y_score_pos = y_score[:, pos_idx] if y_score.ndim > 1 else y_score
                            roc_auc = roc_auc_score(y_true, y_score_pos)
                            mlflow.log_metric("roc_auc", float(roc_auc))
                            fpr, tpr, _ = roc_curve(y_true, y_score_pos)
                            fig, ax = plt.subplots()
                            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                            ax.plot([0, 1], [0, 1], linestyle="--")
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.set_title("ROC Curve")
                            ax.legend()
                            roc_path = out_path.parent / "roc_curve.png"
                            fig.savefig(roc_path)
                            plt.close(fig)
                            mlflow.log_artifact(str(roc_path))
                        else:
                            y_bin = label_binarize(y_true, classes=classes)
                            if y_score.ndim == 1:
                                pass
                            else:
                                fig, ax = plt.subplots()
                                for i, cls in enumerate(classes):
                                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                                    roc_auc = auc(fpr, tpr)
                                    ax.plot(fpr, tpr, label=f"class {cls} (AUC={roc_auc:.2f})")
                                ax.plot([0, 1], [0, 1], linestyle="--")
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Multiclass ROC Curves")
                                ax.legend()
                                mroc_path = out_path.parent / "roc_curve_multiclass.png"
                                fig.savefig(mroc_path)
                                plt.close(fig)
                                mlflow.log_artifact(str(mroc_path))
                except Exception:
                    pass
        except Exception as e:
            print("MLflow logging failed:", e)


if __name__ == "__main__":
    run()
