"""Escalation risk modeling utilities for CivicMind."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _validate_dataframe(df: pd.DataFrame) -> None:
	"""Validate input type and emptiness for complaint datasets."""
	if not isinstance(df, pd.DataFrame):
		raise TypeError("Input must be a pandas DataFrame.")
	if df.empty:
		raise ValueError("Input DataFrame is empty.")


def _get_resolved_flag(df: pd.DataFrame) -> pd.Series:
	"""Return a numeric resolved flag (1 resolved, 0 unresolved) from available columns."""
	if "resolved" in df.columns:
		resolved_raw = df["resolved"]
		if pd.api.types.is_bool_dtype(resolved_raw):
			return resolved_raw.astype(int)
		if pd.api.types.is_numeric_dtype(resolved_raw):
			return (resolved_raw.astype(float) > 0).astype(int)

		resolved_str = resolved_raw.astype(str).str.strip().str.lower()
		return resolved_str.isin({"1", "true", "yes", "resolved"}).astype(int)

	if "status" in df.columns:
		status = df["status"].astype(str).str.strip().str.lower()
		return status.eq("resolved").astype(int)

	raise ValueError("Missing status information. Expected 'resolved' or 'status' column.")


def _compute_days_ago(df: pd.DataFrame) -> pd.Series:
	"""Compute complaint age in days from `date`; return zeros if date is unavailable."""
	if "date" not in df.columns:
		return pd.Series([0] * len(df), index=df.index, dtype=int)

	parsed_dates = pd.to_datetime(df["date"], errors="coerce")
	if parsed_dates.isna().all():
		raise ValueError("Column 'date' could not be parsed into valid dates.")

	today = pd.Timestamp(datetime.now().date())
	days_ago = (today - parsed_dates).dt.days
	return days_ago.fillna(days_ago.median()).clip(lower=0).astype(int)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
	"""Prepare model features and target for escalation prediction.

	Encodes categorical columns (`area`, `category`, `priority`) and creates target:
	`will_escalate = 1` if `(resolved == 0 AND days_to_resolve >= 10)`, else `0`.

	Features returned:
	`area_enc`, `category_enc`, `priority_enc`, `days_ago`, `days_to_resolve`.

	Args:
		df: Input complaints DataFrame.

	Returns:
		Tuple of `(X, y, encoders)` where:
		- `X` is feature DataFrame
		- `y` is target Series
		- `encoders` is a dict of fitted `LabelEncoder`s for categorical fields
	"""
	_validate_dataframe(df)

	required_cols = {"area", "category", "priority", "days_to_resolve"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {sorted(missing)}")

	working = df.copy()
	working["days_to_resolve"] = pd.to_numeric(working["days_to_resolve"], errors="coerce")
	if working["days_to_resolve"].isna().all():
		raise ValueError("Column 'days_to_resolve' has no valid numeric values.")
	working["days_to_resolve"] = working["days_to_resolve"].fillna(
		working["days_to_resolve"].median()
	)

	resolved_flag = _get_resolved_flag(working)
	y = ((resolved_flag == 0) & (working["days_to_resolve"] >= 10)).astype(int)

	encoders: dict[str, LabelEncoder] = {}
	for col in ["area", "category", "priority"]:
		encoder = LabelEncoder()
		working[f"{col}_enc"] = encoder.fit_transform(working[col].astype(str))
		encoders[col] = encoder

	working["days_ago"] = _compute_days_ago(working)

	feature_cols = [
		"area_enc",
		"category_enc",
		"priority_enc",
		"days_ago",
		"days_to_resolve",
	]
	X = working[feature_cols].copy()

	return X, y, encoders


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
	"""Train a Random Forest escalation classifier and print evaluation metrics.

	Uses an 80/20 train-test split and `RandomForestClassifier(n_estimators=100)`.

	Args:
		X: Feature matrix.
		y: Target vector.

	Returns:
		Trained `RandomForestClassifier` model.
	"""
	if not isinstance(X, pd.DataFrame):
		raise TypeError("X must be a pandas DataFrame.")
	if not isinstance(y, (pd.Series, pd.DataFrame)):
		raise TypeError("y must be a pandas Series or DataFrame.")
	if len(X) == 0 or len(y) == 0:
		raise ValueError("X and y cannot be empty.")
	if len(X) != len(y):
		raise ValueError("X and y must have the same number of rows.")

	y_series = y.squeeze()
	if y_series.nunique() < 2:
		raise ValueError("Target y must contain at least two classes for training.")

	stratify = y_series if y_series.nunique() > 1 else None
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y_series,
		test_size=0.2,
		random_state=42,
		stratify=stratify,
	)

	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
	print("Classification Report:")
	print(classification_report(y_test, y_pred, zero_division=0))

	return model


def _safe_label_transform(values: pd.Series, encoder: LabelEncoder) -> pd.Series:
	"""Transform labels using fitted encoder and map unseen values to -1."""
	mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
	return values.astype(str).map(mapping).fillna(-1).astype(int)


def predict_escalation(
	df: pd.DataFrame, model: RandomForestClassifier, encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
	"""Predict escalation probability and risk level for each complaint row.

	Adds:
	- `escalation_prob`: model probability of escalation
	- `risk_level`: Low (0–0.3), Medium (0.3–0.6), High (0.6–1.0)

	Args:
		df: Input complaints DataFrame.
		model: Trained random forest model.
		encoders: Dict of fitted label encoders for `area`, `category`, `priority`.

	Returns:
		Updated DataFrame with prediction columns.
	"""
	_validate_dataframe(df)
	if not hasattr(model, "predict_proba"):
		raise TypeError("Model must support predict_proba().")

	for key in ["area", "category", "priority"]:
		if key not in encoders:
			raise ValueError(f"Missing encoder for '{key}'.")
		if key not in df.columns:
			raise ValueError(f"Missing required column '{key}' in input DataFrame.")

	if "days_to_resolve" not in df.columns:
		raise ValueError("Missing required column 'days_to_resolve' in input DataFrame.")

	out = df.copy()
	out["days_to_resolve"] = pd.to_numeric(out["days_to_resolve"], errors="coerce").fillna(0)
	out["days_ago"] = _compute_days_ago(out)

	out["area_enc"] = _safe_label_transform(out["area"], encoders["area"])
	out["category_enc"] = _safe_label_transform(out["category"], encoders["category"])
	out["priority_enc"] = _safe_label_transform(out["priority"], encoders["priority"])

	feature_cols = [
		"area_enc",
		"category_enc",
		"priority_enc",
		"days_ago",
		"days_to_resolve",
	]

	probs = model.predict_proba(out[feature_cols])
	positive_class_index = list(model.classes_).index(1) if 1 in model.classes_ else -1
	out["escalation_prob"] = probs[:, positive_class_index]

	out["risk_level"] = pd.cut(
		out["escalation_prob"],
		bins=[-0.001, 0.3, 0.6, 1.0],
		labels=["Low", "Medium", "High"],
	).astype(str)

	return out


def get_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
	"""Return sorted feature importance scores for a trained model.

	Args:
		model: Trained `RandomForestClassifier`.
		feature_names: Ordered feature names corresponding to model input columns.

	Returns:
		DataFrame with columns `feature` and `importance`, sorted descending.
	"""
	if not hasattr(model, "feature_importances_"):
		raise TypeError("Model does not expose feature_importances_.")

	importances = model.feature_importances_
	if len(importances) != len(feature_names):
		raise ValueError("Length of feature_names must match model feature importances.")

	return (
		pd.DataFrame({"feature": feature_names, "importance": importances})
		.sort_values("importance", ascending=False)
		.reset_index(drop=True)
	)
