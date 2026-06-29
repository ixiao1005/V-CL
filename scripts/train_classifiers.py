import argparse
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--features_path", required=True)
parser.add_argument("--labels_path", required=True)
parser.add_argument("--label_col", required=True)
parser.add_argument("--output_dir", required=True)
# 在文件开头添加 verbose 参数控制
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()
VERBOSE = args.verbose

mp_path = args.features_path
labels_path = args.labels_path
label_col = args.label_col
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


mp_df = pd.read_csv(mp_path)
labels_df = pd.read_csv(labels_path)[["participant_id", label_col]].copy()

mp_df["participant_id"] = mp_df["participant_id"].astype(str)
labels_df["participant_id"] = labels_df["participant_id"].astype(str)

if mp_df["participant_id"].duplicated().any():
    mp_num = mp_df.select_dtypes(include=[np.number]).columns.tolist()
    mp_df = mp_df.groupby("participant_id")[mp_num].mean().reset_index()

mp_only_df = mp_df.merge(labels_df, on="participant_id", how="inner")

def make_xy(df: pd.DataFrame):
    y = df[label_col].astype(int)
    X = df.drop(columns=["participant_id", label_col]).copy()
    X = X.select_dtypes(include=[np.number])
    return X, y

X_mp, y_mp = make_xy(mp_only_df)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
common_pre = [
    ("imputer", SimpleImputer(strategy="median")),
    ("vt", VarianceThreshold(threshold=1e-8)),
]
models = {}
models["SVM"] = {
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
}

models["DecisionTree"] = {
    "pipe": Pipeline(common_pre + [
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ]),
    "param_grid": {
        "clf__max_depth": [None, 3, 5, 8, 12],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__criterion": ["gini", "entropy"],
    }
}

models["RandomForest"] = {
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


def adapt_param_grid_for_X(param_grid: dict, X: pd.DataFrame) -> dict:
    pg = dict(param_grid)
    if "kbest__k" in pg:
        ks = pg["kbest__k"]
        kept = []
        for k in ks:
            if k == "all":
                kept.append(k)
            elif isinstance(k, (int, np.integer)) and k <= X.shape[1]:
                kept.append(int(k))
        if len(kept) == 0:
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
        return f"{n_feat}→{int(k)}"
    return f"{n_feat}"


def run_models_collect_both(X, y, dataset_name="V-CL",
                            model_names=("SVM", "DecisionTree", "RandomForest", "XGBoost"),
                            verbose_print=False):
    metrics_rows = []
    conf_rows = []

    labels = [0, 1, 2]

    for name in model_names:
        if name not in models:
            print(f"[SKIP] {name} is not configured.")
            continue

        cfg = models[name]
        param_grid = adapt_param_grid_for_X(cfg["param_grid"], X)
        grid = GridSearchCV(
            estimator=cfg["pipe"],
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True
        )
        grid.fit(X, y)
        best_model = grid.best_estimator_

        print(f"Training {name}...")
        if verbose_print:
            print("Best params:", grid.best_params_)
            print("Best CV f1_macro:", grid.best_score_)

        fold_recall_macro = []
        fold_precision_macro = []
        fold_f1_macro = []
        fold_recall_weighted = []
        fold_precision_weighted = []
        fold_f1_weighted = []
        fold_accuracy = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            p_macro, r_macro, f1_macro = prf_avg(y_test, y_pred, average="macro")
            p_w, r_w, f1_w = prf_avg(y_test, y_pred, average="weighted")
            acc = (y_pred == y_test).mean()

            fold_recall_macro.append(r_macro)
            fold_precision_macro.append(p_macro)
            fold_f1_macro.append(f1_macro)
            fold_recall_weighted.append(r_w)
            fold_precision_weighted.append(p_w)
            fold_f1_weighted.append(f1_w)
            fold_accuracy.append(acc)

        recall_macro_mean = np.mean(fold_recall_macro) * 100
        recall_macro_std = np.std(fold_recall_macro, ddof=1) * 100
        precision_macro_mean = np.mean(fold_precision_macro) * 100
        precision_macro_std = np.std(fold_precision_macro, ddof=1) * 100
        f1_macro_mean = np.mean(fold_f1_macro) * 100
        f1_macro_std = np.std(fold_f1_macro, ddof=1) * 100

        recall_weighted_mean = np.mean(fold_recall_weighted) * 100
        recall_weighted_std = np.std(fold_recall_weighted, ddof=1) * 100
        precision_weighted_mean = np.mean(fold_precision_weighted) * 100
        precision_weighted_std = np.std(fold_precision_weighted, ddof=1) * 100
        f1_weighted_mean = np.mean(fold_f1_weighted) * 100
        f1_weighted_std = np.std(fold_f1_weighted, ddof=1) * 100

        acc_mean = np.mean(fold_accuracy) * 100
        acc_std = np.std(fold_accuracy, ddof=1) * 100

        y_pred_oof = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1, method="predict")
        cm = confusion_matrix(y, y_pred_oof, labels=labels)


        if verbose_print:
            print("\n" + "-" * 50)
            print("OOF classification report:")
            print("-" * 50)
            print(classification_report(y, y_pred_oof, target_names=["Low", "Medium", "High"]))
            print("-" * 50)
            print("OOF confusion matrix:")
            print("-" * 50)
            print(cm)
            print("-" * 50)

        feat_desc = get_feature_desc(best_model, X)

        metrics_rows.append({
            "Dataset": dataset_name,
            "Model": name,
            "Features": feat_desc,

            "Recall_macro_mean(%)": recall_macro_mean,
            "Recall_macro_std(%)": recall_macro_std,
            "Precision_macro_mean(%)": precision_macro_mean,
            "Precision_macro_std(%)": precision_macro_std,
            "F1_macro_mean(%)": f1_macro_mean,
            "F1_macro_std(%)": f1_macro_std,

            "Recall_weighted_mean(%)": recall_weighted_mean,
            "Recall_weighted_std(%)": recall_weighted_std,
            "Precision_weighted_mean(%)": precision_weighted_mean,
            "Precision_weighted_std(%)": precision_weighted_std,
            "F1_weighted_mean(%)": f1_weighted_mean,
            "F1_weighted_std(%)": f1_weighted_std,

            "Accuracy_mean(%)": acc_mean,
            "Accuracy_std(%)": acc_std,
        })

        conf_rows.append({
            "Dataset": dataset_name,
            "Model": name,
            "Features": feat_desc,
            "cm00": int(cm[0, 0]), "cm01": int(cm[0, 1]), "cm02": int(cm[0, 2]),
            "cm10": int(cm[1, 0]), "cm11": int(cm[1, 1]), "cm12": int(cm[1, 2]),
            "cm20": int(cm[2, 0]), "cm21": int(cm[2, 1]), "cm22": int(cm[2, 2]),
        })

        print(f"\n Results: F1_macro={f1_macro_mean:.2f}±{f1_macro_std:.2f}%, Acc={acc_mean:.2f}±{acc_std:.2f}%")

    return pd.DataFrame(metrics_rows), pd.DataFrame(conf_rows)

if __name__ == "__main__":
    model_names = ("SVM", "DecisionTree", "RandomForest", "XGBoost")
    task_name = label_col.replace("_label", "")

    print(f"\n Starting analysis: {task_name}")

    df_metrics, df_conf = run_models_collect_both(
        X_mp, y_mp,
        dataset_name="V-CL",
        model_names=model_names,
        verbose_print=VERBOSE
    )


    for col in df_metrics.columns:
        if 'mean' in col.lower() or 'std' in col.lower():
            df_metrics[col] = df_metrics[col].round(2)


    df_metrics.to_csv(os.path.join(output_dir, f"{task_name}_metrics_macro_weighted.csv"), index=False,
                      encoding="utf-8-sig")
    df_conf.to_csv(os.path.join(output_dir, f"{task_name}_confusion_matrices.csv"), index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(os.path.join(output_dir, f"{task_name}_results_metrics_confusion.xlsx"),
                        engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="metrics", index=False)
        df_conf.to_excel(writer, sheet_name="confusion", index=False)

    print(f"\n Results saved to: {output_dir}")

    print("FINAL RESULTS (5-fold CV)")
    print(df_metrics[['Model', 'F1_macro_mean(%)', 'Accuracy_mean(%)']].to_string(index=False))
