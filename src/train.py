from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .feature_engineering import NexHealthFeatureEngineer
from .predict import (
    CALIBRATED_PIPELINE_ARTIFACT_PATH,
    EXPLAINER_PIPELINE_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    SHAP_SUMMARY_ARTIFACT_PATH,
    SHAP_VALUES_ARTIFACT_PATH,
    save_pickle_artifact,
)
from .preprocess import (
    DEFAULT_DATA_PATH,
    DESCRIPTION_COLUMN_MAP,
    GROUP_COLUMN,
    TARGET_COLUMN,
    TRACEABILITY_ONLY_COLUMNS,
    apply_eligibility_filters,
    build_preprocessor,
    load_raw_data,
    prepare_target,
)

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
OUTER_TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CV_SPLITS = 5
PRECISION_FLOOR = 0.60
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)
CALIBRATION_METHOD = "sigmoid"
GWO_EPOCHS = 8
GWO_POP_SIZE = 8
LIGHTGBM_N_JOBS = -1
BASELINE_N_JOBS = 1

LIGHTGBM_IMPORT_ERROR: Exception | None = None
MEALPY_IMPORT_ERROR: Exception | None = None
SHAP_IMPORT_ERROR: Exception | None = None

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - runtime dependency check
    lgb = None
    LIGHTGBM_IMPORT_ERROR = exc

try:
    import shap
except Exception as exc:  # pragma: no cover - runtime dependency check
    shap = None
    SHAP_IMPORT_ERROR = exc

try:
    from mealpy import FloatVar, Problem
    from mealpy.swarm_based.GWO import OriginalGWO
except Exception as exc:  # pragma: no cover - runtime dependency check
    FloatVar = None
    Problem = None
    OriginalGWO = None
    MEALPY_IMPORT_ERROR = exc


BASELINE_MODEL_SPECS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "params": {
            "solver": "liblinear",
            "max_iter": 3000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": BASELINE_N_JOBS,
        },
    },
    "random_forest": {
        "label": "Random Forest",
        "params": {
            "n_estimators": 250,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
            "n_jobs": BASELINE_N_JOBS,
        },
    },
}


if Problem is not None:
    class LightGBMHyperparameterProblem(Problem):
        def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            groups_train: pd.Series,
            precision_floor: float,
            random_state: int,
            bounds: Any,
        ) -> None:
            self.X_train = X_train.reset_index(drop=True)
            self.y_train = y_train.reset_index(drop=True)
            self.groups_train = groups_train.reset_index(drop=True)
            self.precision_floor = precision_floor
            self.random_state = random_state
            super().__init__(bounds=bounds, minmax="max")

        def obj_func(self, solution: np.ndarray) -> float:
            params = decode_lightgbm_solution(solution)
            probabilities = generate_grouped_oof_probabilities(
                model_name="lightgbm",
                model_params=params,
                X_train=self.X_train,
                y_train=self.y_train,
                groups_train=self.groups_train,
            )
            threshold_info = select_best_threshold(
                y_true=self.y_train,
                probabilities=probabilities,
                precision_floor=self.precision_floor,
            )
            return float(threshold_info["selection_score"])
else:
    class LightGBMHyperparameterProblem:  # pragma: no cover - only used when dependency is missing
        def __init__(self, *args, **kwargs) -> None:
            ensure_required_dependencies()


def ensure_required_dependencies() -> None:
    missing = []
    if lgb is None:
        missing.append(f"lightgbm ({LIGHTGBM_IMPORT_ERROR})")
    if OriginalGWO is None or FloatVar is None or Problem is None:
        missing.append(f"mealpy ({MEALPY_IMPORT_ERROR})")
    if shap is None:
        missing.append(f"shap ({SHAP_IMPORT_ERROR})")
    if missing:
        raise ImportError(
            "Missing required dependencies for the LightGBM pipeline: "
            + ", ".join(missing)
            + ". Install them before running `python -m src.train`."
        )


def grouped_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    left_idx, right_idx = next(splitter.split(X, y=y, groups=groups))

    X_left = X.iloc[left_idx].reset_index(drop=True)
    X_right = X.iloc[right_idx].reset_index(drop=True)
    y_left = y.iloc[left_idx].reset_index(drop=True)
    y_right = y.iloc[right_idx].reset_index(drop=True)
    groups_left = groups.iloc[left_idx].reset_index(drop=True)
    groups_right = groups.iloc[right_idx].reset_index(drop=True)
    return X_left, X_right, y_left, y_right, groups_left, groups_right


def build_estimator(model_name: str, model_params: dict):
    params = dict(model_params)
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    if model_name == "lightgbm":
        ensure_required_dependencies()
        return lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_jobs=LIGHTGBM_N_JOBS,
            random_state=params.pop("random_state", RANDOM_STATE),
            verbosity=-1,
            **params,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def build_pipeline(model_name: str, model_params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", NexHealthFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", build_estimator(model_name, model_params)),
        ]
    )


def decode_lightgbm_solution(solution: np.ndarray) -> dict:
    return {
        "random_state": RANDOM_STATE,
        "num_leaves": int(np.clip(np.round(solution[0]), 16, 128)),
        "learning_rate": float(np.clip(solution[1], 0.01, 0.15)),
        "n_estimators": int(np.clip(np.round(solution[2]), 100, 400)),
        "min_child_samples": int(np.clip(np.round(solution[3]), 20, 120)),
        "subsample": float(np.clip(solution[4], 0.70, 1.0)),
        "colsample_bytree": float(np.clip(solution[5], 0.70, 1.0)),
    }


def select_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    precision_floor: float = PRECISION_FLOOR,
) -> dict[str, float | bool]:
    best_floor_candidate = None
    best_fallback_candidate = None

    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f2_score = fbeta_score(y_true, predictions, beta=2, zero_division=0)
        macro_f1 = f1_score(y_true, predictions, average="macro")
        deficit = max(0.0, precision_floor - precision)
        compromise_score = f2_score - deficit
        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f2": float(f2_score),
            "macro_f1": float(macro_f1),
            "meets_floor": precision >= precision_floor,
            "compromise_score": float(compromise_score),
        }

        if candidate["meets_floor"]:
            if best_floor_candidate is None or (candidate["f2"], candidate["precision"], candidate["macro_f1"]) > (
                best_floor_candidate["f2"],
                best_floor_candidate["precision"],
                best_floor_candidate["macro_f1"],
            ):
                best_floor_candidate = candidate

        if best_fallback_candidate is None or (
            candidate["compromise_score"],
            candidate["precision"],
            candidate["f2"],
        ) > (
            best_fallback_candidate["compromise_score"],
            best_fallback_candidate["precision"],
            best_fallback_candidate["f2"],
        ):
            best_fallback_candidate = candidate

    chosen = best_floor_candidate if best_floor_candidate is not None else best_fallback_candidate
    chosen["warning"] = best_floor_candidate is None
    chosen["selection_score"] = float(chosen["f2"] if not chosen["warning"] else chosen["compromise_score"])
    return chosen


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def generate_grouped_oof_probabilities(
    model_name: str,
    model_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> np.ndarray:
    group_kfold = GroupKFold(n_splits=CV_SPLITS)
    probabilities = np.zeros(len(X_train), dtype=float)

    for fold_idx, (fit_idx, valid_idx) in enumerate(group_kfold.split(X_train, y_train, groups_train)):
        fold_params = dict(model_params)
        if "random_state" in fold_params:
            fold_params["random_state"] = RANDOM_STATE + fold_idx
        pipeline = build_pipeline(model_name, fold_params)
        pipeline.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        probabilities[valid_idx] = pipeline.predict_proba(X_train.iloc[valid_idx])[:, 1]

    return probabilities


def tune_lightgbm_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    precision_floor: float = PRECISION_FLOOR,
    random_state: int = RANDOM_STATE,
) -> dict:
    ensure_required_dependencies()
    random.seed(random_state)
    np.random.seed(random_state)

    bounds = FloatVar(
        lb=[16, 0.01, 100, 20, 0.70, 0.70],
        ub=[128, 0.15, 400, 120, 1.00, 1.00],
    )
    problem = LightGBMHyperparameterProblem(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        precision_floor=precision_floor,
        random_state=random_state,
        bounds=bounds,
    )
    optimizer = OriginalGWO(epoch=GWO_EPOCHS, pop_size=GWO_POP_SIZE)
    best = optimizer.solve(problem)
    return decode_lightgbm_solution(np.asarray(best.solution, dtype=float))


def calibrate_model(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    method: str = CALIBRATION_METHOD,
) -> CalibratedClassifierCV:
    group_kfold = GroupKFold(n_splits=CV_SPLITS)
    grouped_splits = list(group_kfold.split(X_train, y_train, groups=groups))
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv=grouped_splits,
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def fit_baseline_result(
    model_name: str,
    model_label: str,
    model_params: dict,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    precision_floor: float,
) -> dict:
    validation_pipeline = build_pipeline(model_name, model_params)
    validation_pipeline.fit(X_fit, y_fit)
    validation_probabilities = validation_pipeline.predict_proba(X_val)[:, 1]
    threshold_info = select_best_threshold(y_val, validation_probabilities, precision_floor=precision_floor)
    validation_metrics = compute_metrics(y_val, validation_probabilities, threshold_info["threshold"])

    final_pipeline = build_pipeline(model_name, model_params)
    final_pipeline.fit(X_train_full, y_train_full)
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_probabilities, threshold_info["threshold"])

    return {
        "name": model_name,
        "label": model_label,
        "params": model_params,
        "threshold_info": threshold_info,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def fit_lightgbm_result(
    best_params: dict,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    groups_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    groups_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    precision_floor: float,
) -> dict:
    validation_calibrated_pipeline = calibrate_model(
        base_pipeline=build_pipeline("lightgbm", best_params),
        X_train=X_fit,
        y_train=y_fit,
        groups=groups_fit,
    )
    validation_probabilities = validation_calibrated_pipeline.predict_proba(X_val)[:, 1]
    threshold_info = select_best_threshold(y_val, validation_probabilities, precision_floor=precision_floor)
    validation_metrics = compute_metrics(y_val, validation_probabilities, threshold_info["threshold"])

    explainer_pipeline = build_pipeline("lightgbm", best_params)
    explainer_pipeline.fit(X_train_full, y_train_full)

    calibrated_pipeline = calibrate_model(
        base_pipeline=build_pipeline("lightgbm", best_params),
        X_train=X_train_full,
        y_train=y_train_full,
        groups=groups_train_full,
    )
    test_probabilities = calibrated_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_probabilities, threshold_info["threshold"])

    return {
        "name": "lightgbm",
        "label": "Calibrated LightGBM",
        "params": best_params,
        "threshold_info": threshold_info,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_probabilities": test_probabilities,
        "calibrated_pipeline": calibrated_pipeline,
        "explainer_pipeline": explainer_pipeline,
    }


def transform_for_shap(explainer_pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    engineered = explainer_pipeline.named_steps["feature_engineering"].transform(X)
    transformed = explainer_pipeline.named_steps["preprocessor"].transform(engineered)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    return pd.DataFrame(transformed, index=engineered.index)


def normalize_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[-1])
    else:
        shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[..., -1]
    return shap_array


def generate_shap_artifacts(explainer_pipeline: Pipeline, X_explain: pd.DataFrame) -> pd.Series:
    ensure_required_dependencies()
    transformed_frame = transform_for_shap(explainer_pipeline, X_explain)
    raw_model = explainer_pipeline.named_steps["model"]
    tree_model = getattr(raw_model, "booster_", raw_model)
    tree_explainer = shap.TreeExplainer(tree_model)
    shap_values = normalize_shap_values(tree_explainer.shap_values(transformed_frame))

    SHAP_VALUES_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(SHAP_VALUES_ARTIFACT_PATH, shap_values)

    shap.summary_plot(shap_values, transformed_frame, show=False)
    plt.savefig(SHAP_SUMMARY_ARTIFACT_PATH, bbox_inches="tight")
    plt.close()

    feature_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=transformed_frame.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)
    return feature_importance


def save_training_artifacts(calibrated_pipeline: CalibratedClassifierCV, explainer_pipeline: Pipeline) -> None:
    preprocessor_pipeline = Pipeline(
        steps=[
            ("feature_engineering", explainer_pipeline.named_steps["feature_engineering"]),
            ("preprocessor", explainer_pipeline.named_steps["preprocessor"]),
        ]
    )
    save_pickle_artifact(calibrated_pipeline, CALIBRATED_PIPELINE_ARTIFACT_PATH)
    save_pickle_artifact(explainer_pipeline, EXPLAINER_PIPELINE_ARTIFACT_PATH)
    save_pickle_artifact(preprocessor_pipeline, PREPROCESSOR_ARTIFACT_PATH)


def print_model_input_sanity(explainer_pipeline: Pipeline, X_train: pd.DataFrame) -> None:
    model_inputs = explainer_pipeline.named_steps["feature_engineering"].transform(X_train.head(5))
    input_columns = model_inputs.columns.tolist()
    description_columns = list(DESCRIPTION_COLUMN_MAP.values())
    raw_id_columns = [column for column in TRACEABILITY_ONLY_COLUMNS if column in input_columns]

    print("\nModel input columns before one-hot encoding:")
    print(input_columns)
    print(f"Description columns present: {all(column in input_columns for column in description_columns)}")
    print(f"Raw ID columns used directly as predictors: {raw_id_columns}")


def print_metric_block(title: str, metrics: dict[str, float], threshold: float) -> None:
    print(f"\n{title}")
    print(f"Threshold: {threshold:.3f}")
    print(f"F2 Score: {metrics['f2']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


def print_comparison_table(title: str, results: list[dict], metric_key: str) -> None:
    print(f"\n{title}")
    for result in results:
        metrics = result[metric_key]
        threshold = result["threshold_info"]["threshold"]
        print(
            f"{result['label']}: "
            f"F2={metrics['f2']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"MacroF1={metrics['macro_f1']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"Threshold={threshold:.3f}"
        )


def print_shap_ranking(feature_importance: pd.Series) -> None:
    print("\nSHAP feature importance ranking:")
    print(feature_importance.to_string())


def main(
    data_path: Path | str = DEFAULT_DATA_PATH,
    precision_floor: float = PRECISION_FLOOR,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ensure_required_dependencies()

    LOGGER.info("Loading raw data from %s", data_path)
    raw_df = load_raw_data(data_path)

    LOGGER.info("Applying deterministic filtering and target preparation")
    prepared_df = prepare_target(apply_eligibility_filters(raw_df))
    X_full = prepared_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    y_full = prepared_df[TARGET_COLUMN].reset_index(drop=True)
    groups_full = X_full[GROUP_COLUMN].reset_index(drop=True)

    LOGGER.info("Creating grouped train/test split")
    X_train_full, X_test, y_train_full, y_test, groups_train_full, groups_test = grouped_holdout_split(
        X=X_full,
        y=y_full,
        groups=groups_full,
        test_size=OUTER_TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    if set(groups_train_full).intersection(set(groups_test)):
        raise ValueError("Group leakage detected between training and test splits.")

    LOGGER.info("Creating grouped fit/validation split from training data")
    X_fit, X_val, y_fit, y_val, groups_fit, groups_val = grouped_holdout_split(
        X=X_train_full,
        y=y_train_full,
        groups=groups_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE + 1,
    )
    if set(groups_fit).intersection(set(groups_val)):
        raise ValueError("Group leakage detected between fit and validation splits.")

    LOGGER.info("Running GWO hyperparameter tuning for LightGBM on fit data only")
    best_params = tune_lightgbm_hyperparameters(
        X_train=X_fit,
        y_train=y_fit,
        groups_train=groups_fit,
        precision_floor=precision_floor,
        random_state=RANDOM_STATE,
    )
    print("Best GWO hyperparameters:")
    print(best_params)

    LOGGER.info("Training and evaluating baseline models")
    results = []
    for model_name, spec in BASELINE_MODEL_SPECS.items():
        results.append(
            fit_baseline_result(
                model_name=model_name,
                model_label=spec["label"],
                model_params=spec["params"],
                X_fit=X_fit,
                y_fit=y_fit,
                X_val=X_val,
                y_val=y_val,
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
                precision_floor=precision_floor,
            )
        )

    LOGGER.info("Training calibrated LightGBM and explainer LightGBM pipelines")
    lightgbm_result = fit_lightgbm_result(
        best_params=best_params,
        X_fit=X_fit,
        y_fit=y_fit,
        groups_fit=groups_fit,
        X_val=X_val,
        y_val=y_val,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        groups_train_full=groups_train_full,
        X_test=X_test,
        y_test=y_test,
        precision_floor=precision_floor,
    )
    results.append(lightgbm_result)

    print_model_input_sanity(lightgbm_result["explainer_pipeline"], X_train_full)
    print_comparison_table("Validation comparison", results, "validation_metrics")
    print_comparison_table("Final untouched test comparison", results, "test_metrics")

    selected_threshold = lightgbm_result["threshold_info"]["threshold"]
    print(f"\nSelected threshold: {selected_threshold:.3f}")
    if lightgbm_result["threshold_info"]["warning"]:
        print(
            f"Warning: no threshold met the precision floor of {precision_floor:.2f}. "
            "Using the best available compromise."
        )

    print_metric_block(
        "Final untouched test metrics - Calibrated LightGBM",
        lightgbm_result["test_metrics"],
        selected_threshold,
    )
    test_predictions = (lightgbm_result["test_probabilities"] >= selected_threshold).astype(int)
    print("\nClassification report - Calibrated LightGBM:")
    print(classification_report(y_test, test_predictions, digits=4))

    LOGGER.info("Generating SHAP artifacts from the explainer pipeline")
    feature_importance = generate_shap_artifacts(
        explainer_pipeline=lightgbm_result["explainer_pipeline"],
        X_explain=X_test,
    )
    print_shap_ranking(feature_importance)

    LOGGER.info("Saving calibrated, explainer, and preprocessor artifacts")
    save_training_artifacts(
        calibrated_pipeline=lightgbm_result["calibrated_pipeline"],
        explainer_pipeline=lightgbm_result["explainer_pipeline"],
    )


if __name__ == "__main__":
    main()
