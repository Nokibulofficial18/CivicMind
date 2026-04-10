from __future__ import annotations

import pandas as pd


def get_area_hotspots(df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate hotspot scores per area based on complaint volume and severity.

	The hotspot score is defined as:
	hotspot_score = (total * 0.4) + (high_priority * 0.6)

	Args:
		df: Input complaints DataFrame. Expected columns include `area` and `priority`.

	Returns:
		A sorted DataFrame with columns:
		`area`, `total`, `high_priority`, `hotspot_score`.
		Rows are sorted by `hotspot_score` in descending order.
	"""
	total = df.groupby("area").size().rename("total")
	high_priority = (
		df[df["priority"].astype(str).str.lower() == "high"]
		.groupby("area")
		.size()
		.rename("high_priority")
	)

	hotspots = (
		pd.concat([total, high_priority], axis=1)
		.fillna(0)
		.reset_index()
		.rename(columns={"index": "area"})
	)

	hotspots["total"] = hotspots["total"].astype(int)
	hotspots["high_priority"] = hotspots["high_priority"].astype(int)
	hotspots["hotspot_score"] = (hotspots["total"] * 0.4) + (
		hotspots["high_priority"] * 0.6
	)

	return hotspots[["area", "total", "high_priority", "hotspot_score"]].sort_values(
		by="hotspot_score", ascending=False
	)


def get_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
	"""Return complaint category counts per area in pivot-table format.

	Args:
		df: Input complaints DataFrame. Expected columns include `area` and `category`.

	Returns:
		A pivoted DataFrame where rows are areas, columns are categories,
		and values are complaint counts. Missing combinations are filled with 0.
	"""
	distribution = pd.pivot_table(
		df,
		index="area",
		columns="category",
		values="complaint_id" if "complaint_id" in df.columns else "area",
		aggfunc="count",
		fill_value=0,
	)

	return distribution.astype(int)


def get_trend(df: pd.DataFrame) -> pd.DataFrame:
	"""Return daily complaint counts per area for the most recent 30 days.

	Args:
		df: Input complaints DataFrame. Expected columns include `date` and `area`.

	Returns:
		A DataFrame containing daily counts grouped by `date` and `area`
		for the last 30 days. Includes columns: `date`, `area`, `count`.
	"""
	trend_df = df.copy()
	trend_df["date"] = pd.to_datetime(trend_df["date"]).dt.date

	latest_date = trend_df["date"].max()
	start_date = latest_date - pd.Timedelta(days=29)

	filtered = trend_df[(trend_df["date"] >= start_date) & (trend_df["date"] <= latest_date)]

	trend = (
		filtered.groupby(["date", "area"])
		.size()
		.reset_index(name="count")
		.sort_values(["date", "area"])
	)

	return trend


def get_hotspot_root_causes(df: pd.DataFrame, hotspot_df: pd.DataFrame) -> pd.DataFrame:
	"""Generate short root-cause explanations for hotspot areas.

	Analysis dimensions per area:
	- complaint growth trend (recent 14 days vs prior 14 days)
	- unresolved ratio
	- dominant complaint category

	Args:
		df: Input complaints DataFrame.
		hotspot_df: Hotspot ranking DataFrame from `get_area_hotspots`.

	Returns:
		DataFrame with one row per hotspot area and columns:
		`area`, `growth_rate_pct`, `unresolved_ratio`, `dominant_category`,
		`dominant_share_pct`, `root_cause`.
	"""
	if df.empty or hotspot_df.empty:
		return pd.DataFrame(
			columns=[
				"area",
				"growth_rate_pct",
				"unresolved_ratio",
				"dominant_category",
				"dominant_share_pct",
				"root_cause",
			]
		)

	working = df.copy()
	working["date"] = pd.to_datetime(working["date"], errors="coerce")
	working = working.dropna(subset=["date", "area", "category"])
	if working.empty:
		return pd.DataFrame(
			columns=[
				"area",
				"growth_rate_pct",
				"unresolved_ratio",
				"dominant_category",
				"dominant_share_pct",
				"root_cause",
			]
		)

	if "status" in working.columns:
		unresolved_mask = working["status"].astype(str).str.lower().eq("unresolved")
	elif "resolved" in working.columns:
		unresolved_mask = pd.to_numeric(working["resolved"], errors="coerce").fillna(0).eq(0)
	else:
		unresolved_mask = pd.Series([False] * len(working), index=working.index)

	max_date = working["date"].max().normalize()
	recent_start = max_date - pd.Timedelta(days=13)
	prior_start = recent_start - pd.Timedelta(days=14)
	prior_end = recent_start - pd.Timedelta(days=1)

	recent_df = working[(working["date"] >= recent_start) & (working["date"] <= max_date)]
	prior_df = working[(working["date"] >= prior_start) & (working["date"] <= prior_end)]

	recent_counts = recent_df.groupby("area").size().rename("recent_count")
	prior_counts = prior_df.groupby("area").size().rename("prior_count")

	results: list[dict[str, object]] = []
	hotspot_areas = hotspot_df["area"].astype(str).tolist()

	for area in hotspot_areas:
		area_df = working[working["area"].astype(str) == area]
		if area_df.empty:
			continue

		total = int(len(area_df))
		unresolved = int(unresolved_mask[area_df.index].sum())
		unresolved_ratio = (unresolved / total) if total > 0 else 0.0

		category_counts = area_df["category"].astype(str).value_counts()
		dominant_category = str(category_counts.index[0]) if not category_counts.empty else "unknown"
		dominant_share = (int(category_counts.iloc[0]) / total) if total > 0 and not category_counts.empty else 0.0

		recent_count = int(recent_counts.get(area, 0))
		prior_count = int(prior_counts.get(area, 0))
		growth_rate = (
			((recent_count - prior_count) / prior_count) * 100
			if prior_count > 0
			else (100.0 if recent_count > 0 else 0.0)
		)

		trend_text = (
			f"complaint volume is increasing ({growth_rate:.1f}% vs previous 14 days)"
			if growth_rate > 0
			else f"complaint volume is stable/declining ({growth_rate:.1f}% vs previous 14 days)"
		)

		root_cause = (
			f"{area} is a hotspot because {trend_text}, unresolved ratio is "
			f"{unresolved_ratio * 100:.1f}%, and {dominant_category} dominates "
			f"({dominant_share * 100:.1f}% of local complaints)."
		)

		results.append(
			{
				"area": area,
				"growth_rate_pct": round(growth_rate, 2),
				"unresolved_ratio": round(unresolved_ratio, 4),
				"dominant_category": dominant_category,
				"dominant_share_pct": round(dominant_share * 100, 2),
				"root_cause": root_cause,
			}
		)

	return pd.DataFrame(results)


def get_predicted_high_risk_areas(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
	"""Forecast likely high-risk areas for the next 7 days.

	Forecast signal combines:
	- trend growth (recent 7 days vs previous 7 days)
	- unresolved complaint ratio

	Args:
		df: Input complaints DataFrame.
		top_n: Number of top forecasted high-risk areas to return.

	Returns:
		DataFrame with columns:
		`area`, `growth_rate_pct`, `unresolved_ratio`, `prediction_score`.
	"""
	if df.empty:
		return pd.DataFrame(
			columns=["area", "growth_rate_pct", "unresolved_ratio", "prediction_score"]
		)

	working = df.copy()
	working["date"] = pd.to_datetime(working["date"], errors="coerce")
	working = working.dropna(subset=["date", "area"])
	if working.empty:
		return pd.DataFrame(
			columns=["area", "growth_rate_pct", "unresolved_ratio", "prediction_score"]
		)

	if "status" in working.columns:
		unresolved_mask = working["status"].astype(str).str.lower().eq("unresolved")
	elif "resolved" in working.columns:
		unresolved_mask = pd.to_numeric(working["resolved"], errors="coerce").fillna(0).eq(0)
	else:
		unresolved_mask = pd.Series([False] * len(working), index=working.index)

	max_date = working["date"].max().normalize()
	recent_start = max_date - pd.Timedelta(days=6)
	prior_start = recent_start - pd.Timedelta(days=7)
	prior_end = recent_start - pd.Timedelta(days=1)

	recent_df = working[(working["date"] >= recent_start) & (working["date"] <= max_date)]
	prior_df = working[(working["date"] >= prior_start) & (working["date"] <= prior_end)]

	recent_counts = recent_df.groupby("area").size().rename("recent_count")
	prior_counts = prior_df.groupby("area").size().rename("prior_count")
	total_counts = working.groupby("area").size().rename("total_count")
	unresolved_counts = working[unresolved_mask].groupby("area").size().rename("unresolved_count")

	forecast = pd.concat([recent_counts, prior_counts, total_counts, unresolved_counts], axis=1).fillna(0)
	forecast = forecast.reset_index().rename(columns={"index": "area"})

	forecast["growth_rate_pct"] = forecast.apply(
		lambda r: ((r["recent_count"] - r["prior_count"]) / r["prior_count"] * 100)
		if r["prior_count"] > 0
		else (100.0 if r["recent_count"] > 0 else 0.0),
		axis=1,
	)
	forecast["unresolved_ratio"] = forecast.apply(
		lambda r: (r["unresolved_count"] / r["total_count"]) if r["total_count"] > 0 else 0.0,
		axis=1,
	)

	# Simple weighted forecasting score.
	forecast["prediction_score"] = (
		(forecast["growth_rate_pct"].clip(lower=0) * 0.45) + (forecast["unresolved_ratio"] * 100 * 0.55)
	)

	return forecast[["area", "growth_rate_pct", "unresolved_ratio", "prediction_score"]].sort_values(
		"prediction_score", ascending=False
	).head(top_n)
