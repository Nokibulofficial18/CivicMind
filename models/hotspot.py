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
