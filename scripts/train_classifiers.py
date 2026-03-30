import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
common_pre = [
    ("imputer", SimpleImputer(strategy="median")),
    ("vt", VarianceThreshold(threshold=1e-8)),
]
models = {
    "SVM_RBF": {
        "pipe": Pipeline(common_pre + [
            ("scaler", StandardScaler()),
            ("kbest", SelectKBest(score_func=f_classif)),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
        ]),
        "param_grid": {
            "kbest__k": [10, 20, 30, 50, 80, 120, "all"],
            "clf__C": [1e-2, 1e-1, 1, 10, 100, 1000],
            "clf__gamma": ["scale", 1e-4, 1e-3, 1e-2, 1e-1],
        }
    },
    "DecisionTree": {
        "pipe": Pipeline(common_pre + [
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
        ]),
        "param_grid": {
            "clf__max_depth": [None, 3, 5, 8, 12],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__criterion": ["gini", "entropy"],
        }
    },
    "RandomForest": {
        "pipe": Pipeline(common_pre + [
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
        ]),
        "param_grid": {
            "clf__n_estimators": [300, 500, 800],
            "clf__max_depth": [None, 6, 10, 16],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", 0.5],
        }
    },
}

if XGBClassifier is not None:
    models["XGBoost"] = {
        "pipe": Pipeline(common_pre + [
            ("clf", XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
                tree_method="hist"
            ))
        ]),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_lambda": [1.0, 5.0],
        }
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate traditional machine learning classifiers using video-level features."
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the input feature CSV file."
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to the labels CSV or TSV file."
    )
    parser.add_argument(
        "--label_col",
        type=str,
        required=True,
        help="Target label column name, e.g. ICL_label, ECL_label, or GCL_label."
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="participant_id",
        help="ID column name in the label file."
    )
    parser.add_argument(
        "--feature_id_col",
        type=str,
        default="participant_id",
        help="ID column name in the feature file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for saving outputs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["SVM_RBF", "DecisionTree", "RandomForest", "XGBoost"],
        help="Models to run."
    )
    return parser.parse_args()

def load_labels_file(labels_path):
    try:
        return pd.read_csv(labels_path)
    except Exception:
        return pd.read_csv(labels_path, sep="\t")

def load_and_prepare_data(
    features_path,
    labels_path,
    label_col,
    id_col="participant_id",
    feature_id_col="participant_id"
):
    feature_df = pd.read_csv(features_path)
    labels_df = load_labels_file(labels_path)

    if feature_id_col not in feature_df.columns:
        raise ValueError(f"Feature file must contain ID column: {feature_id_col}")
    if id_col not in labels_df.columns:
        raise ValueError(f"Labels file must contain ID column: {id_col}")
    if label_col not in labels_df.columns:
        raise ValueError(f"Labels file must contain target label column: {label_col}")

    feature_df[feature_id_col] = feature_df[feature_id_col].astype(str)
    labels_df[id_col] = labels_df[id_col].astype(str)

    if feature_df[feature_id_col].duplicated().any():
        num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_df = feature_df.groupby(feature_id_col)[num_cols].mean().reset_index()

    merged_df = feature_df.merge(
        labels_df[[id_col, label_col]],
        left_on=feature_id_col,
        right_on=id_col,
        how="inner"
    )
    X = merged_df.drop(columns=[feature_id_col, id_col, label_col], errors="ignore").copy()
    X = X.select_dtypes(include=[np.number])
    y = merged_df[label_col].astype(int)
    return X, y

def adapt_param_grid_for_X(param_grid, X):
    pg = dict(param_grid)
    if "kbest__k" in pg:
        ks = pg["kbest__k"]
        kept = []
        for k in ks:
            if k == "all":
                kept.append(k)
            elif isinstance(k, (int, np.integer)) and k <= X.shape[1]:
                kept.append(int(k))
        if not kept:
            kept = ["all"]
        pg["kbest__k"] = sorted(set(kept), key=lambda v: (10**9 if v == "all" else v))
    return pg

def prf_avg(y_true, y_pred, average="macro"):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return p, r, f1

def get_feature_desc(best_estimator, X):
    n_feat = X.shape[1]
    params = best_estimator.get_params()
    if "kbest__k" in params:
        k = params["kbest__k"]
        if k == "all" or k is None:
            return f"{n_feat}"
        return f"{n_feat}->{int(k)}"
    return f"{n_feat}"

def run_models_collect_metrics(
    X,
    y,
    dataset_name="FeatureSet",
    model_names=("SVM_RBF", "DecisionTree", "RandomForest", "XGBoost"),
):
    metrics_rows = []
    conf_rows = []
    labels = sorted(pd.Series(y).dropna().unique().tolist())
    for name in model_names:
        if name not in models:
            continue
        cfg = models[name]
        param_grid = adapt_param_grid_for_X(cfg["param_grid"], X)
        grid = GridSearchCV(
            estimator=cfg["pipe"],
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X, y)
        best_model = grid.best_estimator_
        y_pred_oof = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1, method="predict")
        cm = confusion_matrix(y, y_pred_oof, labels=labels)
        p_macro, r_macro, f1_macro = prf_avg(y, y_pred_oof, average="macro")
        p_weighted, r_weighted, f1_weighted = prf_avg(y, y_pred_oof, average="weighted")
        scores = cross_validate(
            best_model,
            X,
            y,
            cv=cv,
            scoring=["f1_macro", "accuracy"],
            n_jobs=-1,
            return_train_score=False,
        )
        feat_desc = get_feature_desc(best_model, X)
        metrics_rows.append({
            "Dataset": dataset_name,
            "Model": name,
            "Features": feat_desc,
            "BestParams": str(grid.best_params_),
            "BestCV_F1_macro": grid.best_score_,
            "Recall_macro(%)": r_macro * 100,
            "Precision_macro(%)": p_macro * 100,
            "F1_macro(%)": f1_macro * 100,
            "Recall_weighted(%)": r_weighted * 100,
            "Precision_weighted(%)": p_weighted * 100,
            "F1_weighted(%)": f1_weighted * 100,
            "MacroF1_mean(5fold)": scores["test_f1_macro"].mean(),
            "MacroF1_std(5fold)": scores["test_f1_macro"].std(ddof=1),
            "Acc_mean(5fold)": scores["test_accuracy"].mean(),
        })
        conf_row = {
            "Dataset": dataset_name,
            "Model": name,
            "Features": feat_desc,
        }
        for i in range(len(labels)):
            for j in range(len(labels)):
                conf_row[f"cm{i}{j}"] = int(cm[i, j])
        conf_rows.append(conf_row)
    return pd.DataFrame(metrics_rows), pd.DataFrame(conf_rows)

def main():
    args = parse_args()
    features_path = Path(args.features_path)
    labels_path = Path(args.labels_path)
    output_dir = Path(args.output_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_and_prepare_data(
        features_path=features_path,
        labels_path=labels_path,
        label_col=args.label_col,
        id_col=args.id_col,
        feature_id_col=args.feature_id_col,
    )
    df_metrics, df_conf = run_models_collect_metrics(
        X=X,
        y=y,
        dataset_name=args.label_col,
        model_names=tuple(args.models),
    )
    pct_cols = [
        "Recall_macro(%)",
        "Precision_macro(%)",
        "F1_macro(%)",
        "Recall_weighted(%)",
        "Precision_weighted(%)",
        "F1_weighted(%)",
    ]
    for c in pct_cols:
        if c in df_metrics.columns:
            df_metrics[c] = df_metrics[c].round(2)
    round4_cols = ["BestCV_F1_macro", "MacroF1_mean(5fold)", "MacroF1_std(5fold)", "Acc_mean(5fold)"]
    for c in round4_cols:
        if c in df_metrics.columns:
            df_metrics[c] = df_metrics[c].round(4)
    metrics_csv = output_dir / f"{args.label_col}_metrics.csv"
    conf_csv = output_dir / f"{args.label_col}_confusion_matrices.csv"
    results_xlsx = output_dir / f"{args.label_col}_results.xlsx"
    df_metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    df_conf.to_csv(conf_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(results_xlsx, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="metrics", index=False)
        df_conf.to_excel(writer, sheet_name="confusion", index=False)
    print(f"Saved results to: {output_dir}")

if __name__ == "__main__":
    main()