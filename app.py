from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data.generate_data import generate_complaints
from models.escalation import (
	get_feature_importance,
	predict_escalation,
	prepare_features,
	train_model,
)
from models.hotspot import get_area_hotspots, get_category_distribution, get_trend
from utils.map_utils import add_area_markers, generate_heatmap


st.set_page_config(page_title="CivicMind — Dhaka Urban Intelligence", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
	"""Load complaints from CSV if present, otherwise generate synthetic data."""
	data_path = Path("data") / "complaints.csv"
	if data_path.exists():
		df = pd.read_csv(data_path)
	else:
		df = generate_complaints(num_rows=600, seed=42)
		data_path.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(data_path, index=False)

	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce")

	return df


@st.cache_data(show_spinner=False)
def train_escalation_pipeline(df: pd.DataFrame):
	"""Prepare features and train escalation model with cached results."""
	X, y, encoders = prepare_features(df)
	model = train_model(X, y)
	feature_names = list(X.columns)
	return model, encoders, feature_names


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply sidebar filters for area, category, and recent date window."""
	st.sidebar.header("Filters")

	all_areas = sorted(df["area"].dropna().astype(str).unique().tolist())
	all_categories = sorted(df["category"].dropna().astype(str).unique().tolist())

	selected_areas = st.sidebar.multiselect("Areas", options=all_areas, default=all_areas)
	selected_categories = st.sidebar.multiselect(
		"Categories", options=all_categories, default=all_categories
	)
	selected_days = st.sidebar.select_slider(
		"Date range",
		options=[7, 14, 30, 60],
		value=60,
		format_func=lambda x: f"Last {x} days",
	)

	filtered = df.copy()
	if selected_areas:
		filtered = filtered[filtered["area"].isin(selected_areas)]
	else:
		filtered = filtered.iloc[0:0]

	if selected_categories:
		filtered = filtered[filtered["category"].isin(selected_categories)]
	else:
		filtered = filtered.iloc[0:0]

	if not filtered.empty and "date" in filtered.columns:
		max_date = filtered["date"].max()
		if pd.notna(max_date):
			start_date = max_date - pd.Timedelta(days=int(selected_days - 1))
			filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= max_date)]

	return filtered


def style_risk(val: str) -> str:
	"""Return color style for risk_level values."""
	color_map = {
		"High": "#ff4d4f",
		"Medium": "#ffa940",
		"Low": "#73d13d",
	}
	color = color_map.get(str(val), "#d9d9d9")
	return f"background-color: {color}; color: #111; font-weight: 600;"


def build_department_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Map complaint categories to responsible departments."""
	dept_map = {
		"road": "City Corp",
		"waste": "City Corp",
		"water": "WASA",
		"traffic": "Traffic Police",
	}
	out = df.copy()
	out["department"] = out["category"].astype(str).str.lower().map(dept_map).fillna("Other")
	return out


def main() -> None:
	st.title("CivicMind — Dhaka Urban Intelligence Dashboard")

	df = load_data()
	if df.empty:
		st.error("No complaint data available.")
		return

	try:
		model, encoders, feature_names = train_escalation_pipeline(df)
	except Exception as ex:
		st.error(f"Model training failed: {ex}")
		return

	filtered_df = apply_filters(df)
	if filtered_df.empty:
		st.warning("No data available for the selected filters.")
		return

	try:
		pred_df = predict_escalation(filtered_df, model, encoders)
	except Exception as ex:
		st.error(f"Escalation prediction failed: {ex}")
		return

	hotspot_df = get_area_hotspots(pred_df)

	total_complaints = int(len(pred_df))
	high_risk_count = int((pred_df["risk_level"] == "High").sum())
	areas_monitored = int(pred_df["area"].nunique())
	avg_resolution_days = float(pd.to_numeric(pred_df["days_to_resolve"], errors="coerce").mean())

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Total Complaints", f"{total_complaints}")
	c2.metric("High Risk Complaints", f"{high_risk_count}")
	c3.metric("Areas Monitored", f"{areas_monitored}")
	c4.metric("Avg Resolution Days", f"{avg_resolution_days:.2f}")

	tab1, tab2, tab3, tab4 = st.tabs(
		["Hotspot Map", "Escalation Predictor", "Department Workload", "Trends"]
	)

	with tab1:
		st.subheader("Dhaka Complaint Hotspots")
		map_obj = generate_heatmap(pred_df)
		map_obj = add_area_markers(map_obj, hotspot_df)
		st.components.v1.html(map_obj.get_root().render(), height=560)

		hotspot_plot = hotspot_df.sort_values("hotspot_score", ascending=True)
		fig_hotspot = px.bar(
			hotspot_plot,
			x="hotspot_score",
			y="area",
			orientation="h",
			title="Hotspot Score by Area",
			color="hotspot_score",
			color_continuous_scale="Reds",
		)
		st.plotly_chart(fig_hotspot, use_container_width=True)

	with tab2:
		st.subheader("Escalation Risk Overview")

		top_risk = pred_df.sort_values("escalation_prob", ascending=False).head(10).copy()
		cols = ["area", "category", "priority", "days_ago", "escalation_prob", "risk_level"]
		top_risk = top_risk[cols]
		top_risk["escalation_prob"] = top_risk["escalation_prob"].round(3)

		st.dataframe(
			top_risk.style.applymap(style_risk, subset=["risk_level"]),
			use_container_width=True,
			hide_index=True,
		)

		area_prob = (
			pred_df.groupby("area", as_index=False)["escalation_prob"].mean().sort_values(
				"escalation_prob", ascending=False
			)
		)
		fig_prob = px.bar(
			area_prob,
			x="area",
			y="escalation_prob",
			title="Average Escalation Probability by Area",
			color="escalation_prob",
			color_continuous_scale="OrRd",
		)
		st.plotly_chart(fig_prob, use_container_width=True)

		try:
			fi_df = get_feature_importance(model, feature_names)
			fig_fi = px.bar(
				fi_df,
				x="importance",
				y="feature",
				orientation="h",
				title="Model Feature Importance",
			)
			st.plotly_chart(fig_fi, use_container_width=True)
		except Exception:
			pass

	with tab3:
		st.subheader("Department Workload")
		dept_df = build_department_columns(pred_df)

		dept_share = dept_df["department"].value_counts().reset_index()
		dept_share.columns = ["department", "count"]
		fig_pie = px.pie(
			dept_share,
			values="count",
			names="department",
			title="Complaint Share by Department",
			hole=0.35,
		)
		st.plotly_chart(fig_pie, use_container_width=True)

		if "status" in dept_df.columns:
			unresolved_mask = dept_df["status"].astype(str).str.lower().eq("unresolved")
		elif "resolved" in dept_df.columns:
			unresolved_mask = pd.to_numeric(dept_df["resolved"], errors="coerce").fillna(0).eq(0)
		else:
			unresolved_mask = pd.Series([False] * len(dept_df), index=dept_df.index)

		unresolved_dept = (
			dept_df[unresolved_mask]["department"].value_counts().reset_index()
		)
		unresolved_dept.columns = ["department", "unresolved_count"]
		fig_unresolved = px.bar(
			unresolved_dept,
			x="department",
			y="unresolved_count",
			title="Unresolved Complaints per Department",
			color="unresolved_count",
			color_continuous_scale="Sunsetdark",
		)
		st.plotly_chart(fig_unresolved, use_container_width=True)

	with tab4:
		st.subheader("Complaint Trends")

		trend_df = get_trend(pred_df)
		if not trend_df.empty:
			fig_trend = px.line(
				trend_df,
				x="date",
				y="count",
				color="area",
				title="Daily Complaints by Area",
				markers=True,
			)
			st.plotly_chart(fig_trend, use_container_width=True)
		else:
			st.info("Trend data not available for current filters.")

		category_dist = get_category_distribution(pred_df).reset_index()
		category_melt = category_dist.melt(
			id_vars="area", var_name="category", value_name="count"
		)
		fig_stack = px.bar(
			category_melt,
			x="area",
			y="count",
			color="category",
			title="Category Distribution by Area",
		)
		fig_stack.update_layout(barmode="stack")
		st.plotly_chart(fig_stack, use_container_width=True)

	st.markdown("---")
	st.subheader("Today's Action Brief")

	brief_df = (
		pred_df.groupby("area", as_index=False)
		.agg(
			complaint_count=("area", "size"),
			avg_escalation_prob=("escalation_prob", "mean"),
		)
		.sort_values(["avg_escalation_prob", "complaint_count"], ascending=[False, False])
		.head(3)
	)

	if brief_df.empty:
		st.info("No action brief available for current filters.")
	else:
		for _, row in brief_df.iterrows():
			st.error(
				f"{row['area']}: {int(row['complaint_count'])} complaints | "
				f"Avg escalation risk: {row['avg_escalation_prob'] * 100:.1f}%"
			)


if __name__ == "__main__":
	main()
