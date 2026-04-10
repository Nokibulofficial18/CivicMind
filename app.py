"""CivicMind Streamlit dashboard application."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data.generate_data import generate_complaints
from models.escalation import (
	get_feature_importance,
	predict_escalation,
	prepare_features,
	train_model,
)
from models.hotspot import (
	get_area_hotspots,
	get_category_distribution,
	get_hotspot_root_causes,
	get_predicted_high_risk_areas,
	get_trend,
)
from utils.map_utils import add_area_markers, generate_heatmap


st.set_page_config(
	page_title="CivicMind — Dhaka Urban Intelligence",
	page_icon="🏙️",
	layout="wide",
)

APP_TITLE = "CivicMind — Dhaka Urban Intelligence Dashboard"
DATE_WINDOW_OPTIONS = [7, 14, 30, 60]
RISK_COLORS = {"High": "#D7263D", "Medium": "#F18F01", "Low": "#2A9D8F"}
PLOT_TEMPLATE = "plotly_white"
CATEGORICAL_COLUMNS = ["area", "category", "priority", "status"]


AREAS = ["Mirpur", "Dhanmondi", "Uttara", "Farmgate", "Demra", "Gulshan", "Mohammadpur"]
CATEGORIES = ["road", "waste", "water", "traffic"]
PRIORITIES = ["low", "medium", "high"]
USERS = {
	"user1": {"password": "1234", "role": "user"},
	"admin": {"password": "admin123", "role": "admin"},
}

DEPARTMENT_MAP = {
	"road": "City Corporation",
	"waste": "City Corporation",
	"water": "WASA",
	"traffic": "Traffic Police",
}


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
	"""Return an empty DataFrame with same schema."""
	return df.iloc[0:0].copy()


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
	"""Reduce memory usage by applying compact dtypes where appropriate."""
	optimized_df = df.copy()

	for column in CATEGORICAL_COLUMNS:
		if column in optimized_df.columns:
			optimized_df[column] = optimized_df[column].astype("category")

	if "days_to_resolve" in optimized_df.columns:
		optimized_df["days_to_resolve"] = pd.to_numeric(
			optimized_df["days_to_resolve"], errors="coerce"
		).fillna(0).astype("int16")

	if "resolved" in optimized_df.columns:
		optimized_df["resolved"] = pd.to_numeric(optimized_df["resolved"], errors="coerce").fillna(0).astype(
			"int8"
		)

	if "escalation_prob" in optimized_df.columns:
		optimized_df["escalation_prob"] = pd.to_numeric(
			optimized_df["escalation_prob"], errors="coerce"
		).fillna(0).astype("float32")

	return optimized_df


def _inject_custom_styles() -> None:
	"""Inject custom CSS for a polished, demo-ready dashboard appearance."""
	st.markdown(
		"""
		<style>
			.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
			h1 {font-size: 2.1rem !important; font-weight: 800 !important; letter-spacing: 0.2px;}
			h2, h3 {font-size: 1.35rem !important; font-weight: 700 !important;}
			[data-testid="stMetricValue"] {font-size: 1.7rem !important;}
			[data-testid="stMetricLabel"] {font-size: 1rem !important;}
			div[data-testid="stDataFrame"] div[role="table"] {font-size: 0.95rem;}
			section[data-testid="stSidebar"] h2 {font-size: 1.1rem !important;}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _style_figure(
	fig,
	x_title: str,
	y_title: str,
	legend_title: str | None = None,
) -> None:
	"""Apply consistent Plotly styling across dashboard charts."""
	fig.update_layout(
		template=PLOT_TEMPLATE,
		font={"size": 14},
		title={"font": {"size": 22}},
		xaxis_title=x_title,
		yaxis_title=y_title,
		legend_title_text=legend_title,
		hoverlabel={"font_size": 13},
		margin={"l": 20, "r": 20, "t": 60, "b": 40},
	)


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

	return _optimize_dataframe_memory(df)


@st.cache_resource(show_spinner=False)
def train_escalation_pipeline(
	df: pd.DataFrame,
) -> tuple[RandomForestClassifier, dict[str, LabelEncoder], list[str]]:
	"""Prepare features and train escalation model with cached results."""
	X, y, encoders = prepare_features(df)
	model = train_model(X, y)
	feature_names = list(X.columns)
	return model, encoders, feature_names


def _init_session_state() -> None:
	"""Initialize session containers for submissions and prediction feedback."""
	if "submitted_complaints" not in st.session_state:
		st.session_state["submitted_complaints"] = []
	if "last_submission_result" not in st.session_state:
		st.session_state["last_submission_result"] = None
	if "logged_in" not in st.session_state:
		st.session_state["logged_in"] = False
	if "username" not in st.session_state:
		st.session_state["username"] = None
	if "role" not in st.session_state:
		st.session_state["role"] = None
	if "master_df" not in st.session_state:
		st.session_state["master_df"] = load_data()


def _authenticate(username: str, password: str) -> tuple[bool, str | None]:
	"""Validate credentials against the hardcoded user store."""
	user_config = USERS.get(username)
	if not user_config:
		return False, None
	if user_config["password"] != password:
		return False, None
	return True, str(user_config["role"])


def _save_dataset(df: pd.DataFrame) -> None:
	"""Persist current complaint dataset to CSV."""
	output_path = Path("data") / "complaints.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)


def _render_login() -> None:
	"""Render login form and update session auth state."""
	st.title("🔐 CivicMind Login")
	st.caption("Please sign in to access the Urban Intelligence Dashboard.")

	with st.form("login_form", clear_on_submit=False):
		username = st.text_input("Username", placeholder="Enter username")
		password = st.text_input("Password", type="password", placeholder="Enter password")
		login_clicked = st.form_submit_button("Login", use_container_width=True)

	if not login_clicked:
		return

	is_valid, role = _authenticate(username.strip(), password)
	if not is_valid or role is None:
		st.error("Invalid username or password.")
		return

	st.session_state["logged_in"] = True
	st.session_state["username"] = username.strip()
	st.session_state["role"] = role
	st.success("Login successful.")
	st.rerun()


def _logout() -> None:
	"""Clear authentication session and return user to login view."""
	st.session_state["logged_in"] = False
	st.session_state["username"] = None
	st.session_state["role"] = None
	st.rerun()


def _render_user_header() -> None:
	"""Render active user identity, role badge, and logout action."""
	username = st.session_state.get("username", "unknown")
	role = st.session_state.get("role", "user")
	role_badge = "Admin" if role == "admin" else "User"

	header_col_1, header_col_2, header_col_3 = st.columns([3, 2, 1])
	header_col_1.markdown(f"**👤 Logged in user:** {username}")
	header_col_2.markdown(f"**🏷️ Logged in as {role_badge}**")
	if header_col_3.button("Logout", use_container_width=True):
		_logout()


def _build_current_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
	"""Return current master dataset snapshot."""
	return _optimize_dataframe_memory(base_df.copy())


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure `date` column is datetime when available."""
	out = df.copy()
	if "date" in out.columns:
		out["date"] = pd.to_datetime(out["date"], errors="coerce")
	return _optimize_dataframe_memory(out)


@st.cache_data(show_spinner=False)
def _compute_cached_views(predicted_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Cache derived analytics frames to avoid repeated groupby/pivot computations."""
	hotspot_df = get_area_hotspots(predicted_df)
	trend_df = get_trend(predicted_df)
	category_distribution_df = get_category_distribution(predicted_df).reset_index()
	action_brief_df = (
		predicted_df.groupby("area", as_index=False)
		.agg(
			complaint_count=("area", "size"),
			avg_escalation_prob=("escalation_prob", "mean"),
		)
		.sort_values(["avg_escalation_prob", "complaint_count"], ascending=[False, False])
		.head(3)
	)
	return hotspot_df, trend_df, category_distribution_df, action_brief_df


@st.cache_data(show_spinner=False)
def _build_heatmap_html(predicted_df: pd.DataFrame, hotspot_df: pd.DataFrame) -> str:
	"""Cache rendered Folium map HTML to speed up rerenders."""
	map_obj = generate_heatmap(predicted_df)
	map_obj = add_area_markers(map_obj, hotspot_df)
	return map_obj.get_root().render()


def _risk_insight(prob: float, priority: str, days_to_resolve: int) -> str:
	"""Generate a short AI insight for escalation risk explanation."""
	if prob >= 0.6:
		return (
			"This complaint is likely to escalate due to severity signals "
			"and delayed expected resolution."
		)
	if prob >= 0.3:
		if priority == "high" or days_to_resolve >= 10:
			return "Moderate escalation risk detected; close monitoring is recommended."
		return "Risk is moderate and may increase if response is delayed."
	return "Current escalation risk is low with timely intervention potential."


def render_submit_complaint(
	base_df: pd.DataFrame,
	model: RandomForestClassifier,
	encoders: dict[str, LabelEncoder],
) -> None:
	"""Render complaint submission UI and persist submission in session state."""
	st.sidebar.markdown("---")
	st.sidebar.subheader("📝 Submit New Complaint")

	with st.sidebar.form("submit_complaint_form", clear_on_submit=True):
		area = st.selectbox("Area", options=AREAS, index=0, key="submit_area")
		category = st.selectbox("Category", options=CATEGORIES, index=0, key="submit_category")
		priority = st.selectbox("Priority", options=PRIORITIES, index=1, key="submit_priority")
		days_to_resolve = st.slider(
			"Days to Resolve", min_value=1, max_value=20, value=7, key="submit_days_to_resolve"
		)
		text = st.text_area(
			"Complaint description",
			placeholder="Describe the issue briefly (location, impact, urgency).",
			key="submit_text",
		)
		submitted = st.form_submit_button("🚀 Analyze & Submit", use_container_width=True)

	if not submitted:
		return

	complaint_text = text.strip() if text and text.strip() else f"Reported {category} issue in {area}."
	if len(complaint_text) < 10:
		st.sidebar.warning("Please provide a slightly more detailed complaint description.")
		return

	current_dt = pd.Timestamp.now()
	base_count = len(base_df)
	sub_count = len(st.session_state.get("submitted_complaints", [])) + 1
	new_row = {
		"complaint_id": f"SUB-{base_count + sub_count:05d}",
		"date": current_dt,
		"area": area,
		"category": category,
		"priority": priority,
		"resolved": 0,
		"status": "unresolved",
		"days_to_resolve": int(days_to_resolve),
		"complaint_text": complaint_text,
		"text": complaint_text,
	}

	one_row_df = pd.DataFrame([new_row])
	try:
		pred_one = predict_escalation(one_row_df, model, encoders)
		prob = float(pred_one.loc[pred_one.index[0], "escalation_prob"])
		risk = str(pred_one.loc[pred_one.index[0], "risk_level"])
	except Exception as ex:
		st.sidebar.error(f"Prediction failed for submitted complaint: {ex}")
		return

	dept = DEPARTMENT_MAP.get(category, "City Operations")
	insight = _risk_insight(prob, priority, int(days_to_resolve))

	st.session_state["submitted_complaints"].append(new_row)
	master_df = st.session_state.get("master_df", base_df).copy()
	master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)
	st.session_state["master_df"] = _optimize_dataframe_memory(master_df)
	_save_dataset(st.session_state["master_df"])
	st.session_state["last_submission_result"] = {
		"submitted_at": current_dt,
		"area": area,
		"category": category,
		"priority": priority,
		"days_to_resolve": int(days_to_resolve),
		"escalation_prob": prob,
		"risk_level": risk,
		"department": dept,
		"insight": insight,
	}
	st.sidebar.success("Complaint analyzed and submitted successfully.")


def render_submission_result() -> None:
	"""Render latest submission result card and recently submitted table."""
	result = st.session_state.get("last_submission_result")
	if not result:
		return

	risk = str(result["risk_level"])
	prob_pct = float(result["escalation_prob"]) * 100
	dept = result["department"]
	insight = result["insight"]

	if risk == "High":
		st.error(
			f"⚠ HIGH RISK\n\nEscalation Probability: {prob_pct:.1f}%\n\n"
			f"Suggested Department: {dept}\n\nAI Insight: {insight}"
		)
	elif risk == "Medium":
		st.warning(
			f"🟠 MEDIUM RISK\n\nEscalation Probability: {prob_pct:.1f}%\n\n"
			f"Suggested Department: {dept}\n\nAI Insight: {insight}"
		)
	else:
		st.success(
			f"✅ LOW RISK\n\nEscalation Probability: {prob_pct:.1f}%\n\n"
			f"Suggested Department: {dept}\n\nAI Insight: {insight}"
		)

	recent = st.session_state.get("submitted_complaints", [])
	if recent:
		recent_df = pd.DataFrame(recent).copy()
		show_cols = [
			"date",
			"area",
			"category",
			"priority",
			"days_to_resolve",
			"status",
			"complaint_text",
		]
		show_cols = [c for c in show_cols if c in recent_df.columns]
		recent_df = recent_df[show_cols].sort_values("date", ascending=False).head(8)
		st.markdown("#### 🧾 Recent Submissions")
		st.dataframe(recent_df, width="stretch", hide_index=True)


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
		options=DATE_WINDOW_OPTIONS,
		value=60,
		format_func=lambda x: f"Last {x} days",
	)

	if not selected_areas or not selected_categories:
		return _empty_like(df)

	filtered = df[
		df["area"].isin(selected_areas) & df["category"].isin(selected_categories)
	].copy()

	if not filtered.empty and "date" in filtered.columns:
		max_date = filtered["date"].max()
		if pd.notna(max_date):
			start_date = max_date - pd.Timedelta(days=int(selected_days - 1))
			filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= max_date)]

	return filtered


def style_risk(val: str) -> str:
	"""Return color style for risk_level values."""
	color = RISK_COLORS.get(str(val), "#d9d9d9")
	text_color = "#ffffff" if str(val) in {"High", "Medium"} else "#111111"
	return f"background-color: {color}; color: {text_color}; font-weight: 700;"


def build_department_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Map complaint categories to responsible departments."""
	out = df.copy()
	out["department"] = out["category"].astype(str).str.lower().map(DEPARTMENT_MAP).fillna("Other")
	return out


def _render_metrics(df: pd.DataFrame) -> None:
	"""Render the top KPI metric cards."""
	st.markdown("### 📊 City Operations Snapshot")
	total_complaints = int(len(df))
	high_risk_count = int((df["risk_level"] == "High").sum())
	areas_monitored = int(df["area"].nunique())
	avg_resolution_days = float(pd.to_numeric(df["days_to_resolve"], errors="coerce").mean())

	col_1, col_2, col_3, col_4 = st.columns(4)
	col_1.metric("Total Complaints", f"{total_complaints}")
	col_2.metric("High Risk Complaints", f"{high_risk_count}")
	col_3.metric("Areas Monitored", f"{areas_monitored}")
	col_4.metric("Avg Resolution Days", f"{avg_resolution_days:.2f}")
	st.markdown(" ")


def _render_hotspot_tab(df: pd.DataFrame, hotspots: pd.DataFrame) -> None:
	"""Render map and hotspot ranking visuals."""
	st.subheader("🗺️ Dhaka Complaint Hotspots")

	try:
		heatmap_html = _build_heatmap_html(df, hotspots)
		st.components.v1.html(heatmap_html, height=560)
	except Exception as exc:
		st.warning(f"Map rendering unavailable: {exc}")

	hotspot_plot = hotspots.sort_values("hotspot_score", ascending=True)
	fig_hotspot = px.bar(
		hotspot_plot,
		x="hotspot_score",
		y="area",
		orientation="h",
		title="Hotspot Score by Area",
		color="hotspot_score",
		color_continuous_scale="Reds",
		labels={"hotspot_score": "Hotspot Score", "area": "Area"},
		hover_data={"hotspot_score": ":.2f", "area": True},
	)
	_style_figure(fig_hotspot, x_title="Hotspot Score", y_title="Area")
	fig_hotspot.update_traces(hovertemplate="Area: %{y}<br>Hotspot Score: %{x:.2f}<extra></extra>")
	st.plotly_chart(fig_hotspot, width="stretch")

	root_cause_df = get_hotspot_root_causes(df, hotspots)
	if not root_cause_df.empty:
		st.markdown("### 🧭 Root Cause Analysis")
		for _, row in root_cause_df.iterrows():
			st.info(f"**{row['area']}** — {row['root_cause']}")


def _render_escalation_tab(
	df: pd.DataFrame,
	model: RandomForestClassifier,
	feature_names: list[str],
) -> None:
	"""Render escalation risk table, area-level risk chart, and feature importance."""
	st.subheader("⚠️ Escalation Risk Overview")

	top_risk_df = df.sort_values("escalation_prob", ascending=False).head(10).copy()
	visible_columns = ["area", "category", "priority", "days_ago", "escalation_prob", "risk_level"]
	top_risk_df = top_risk_df[visible_columns]
	top_risk_df["escalation_prob"] = top_risk_df["escalation_prob"].round(3)

	st.dataframe(
		top_risk_df.style.applymap(style_risk, subset=["risk_level"]),
		width="stretch",
		hide_index=True,
	)

	area_risk_df = (
		df.groupby("area", as_index=False)["escalation_prob"]
		.mean()
		.sort_values("escalation_prob", ascending=False)
	)
	area_risk_df["risk_level"] = pd.cut(
		area_risk_df["escalation_prob"],
		bins=[-0.001, 0.3, 0.6, 1.0],
		labels=["Low", "Medium", "High"],
	).astype(str)
	fig_prob = px.bar(
		area_risk_df,
		x="area",
		y="escalation_prob",
		title="Average Escalation Probability by Area",
		color="risk_level",
		color_discrete_map=RISK_COLORS,
		labels={
			"area": "Area",
			"escalation_prob": "Avg Escalation Probability",
			"risk_level": "Risk Level",
		},
		hover_data={"escalation_prob": ":.2%", "risk_level": True},
	)
	_style_figure(fig_prob, x_title="Area", y_title="Avg Escalation Probability", legend_title="Risk")
	fig_prob.update_traces(
		hovertemplate="Area: %{x}<br>Escalation Prob: %{y:.2%}<br>Risk: %{fullData.name}<extra></extra>"
	)
	st.plotly_chart(fig_prob, width="stretch")

	try:
		feature_importance_df = get_feature_importance(model, feature_names)
		fig_feature_importance = px.bar(
			feature_importance_df,
			x="importance_score",
			y="feature_name",
			orientation="h",
			title="Model Feature Importance",
			labels={"importance_score": "Importance Score", "feature_name": "Model Feature"},
			hover_data={"importance_score": ":.4f", "feature_name": True},
		)
		_style_figure(fig_feature_importance, x_title="Importance Score", y_title="Feature")
		fig_feature_importance.update_traces(
			hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>"
		)
		st.plotly_chart(fig_feature_importance, width="stretch")
	except Exception:
		pass


def _resolve_unresolved_mask(df: pd.DataFrame) -> pd.Series:
	"""Build unresolved mask using available schema variants."""
	if "status" in df.columns:
		return df["status"].astype(str).str.lower().eq("unresolved")
	if "resolved" in df.columns:
		return pd.to_numeric(df["resolved"], errors="coerce").fillna(0).eq(0)
	return pd.Series([False] * len(df), index=df.index)


def _render_department_tab(df: pd.DataFrame) -> None:
	"""Render department workload charts."""
	st.subheader("🏛️ Department Workload")
	department_df = build_department_columns(df)

	department_share_df = department_df["department"].value_counts().reset_index()
	department_share_df.columns = ["department", "count"]
	department_colors = {
		"City Corporation": "#457B9D",
		"WASA": "#2A9D8F",
		"Traffic Police": "#F4A261",
		"Other": "#8D99AE",
	}
	fig_pie = px.pie(
		department_share_df,
		values="count",
		names="department",
		title="Complaint Share by Department",
		hole=0.35,
		color="department",
		color_discrete_map=department_colors,
		labels={"count": "Complaints", "department": "Department"},
	)
	_style_figure(fig_pie, x_title="", y_title="", legend_title="Department")
	fig_pie.update_traces(hovertemplate="%{label}: %{value} complaints (%{percent})<extra></extra>")
	st.plotly_chart(fig_pie, width="stretch")

	unresolved_mask = _resolve_unresolved_mask(department_df)
	unresolved_df = department_df[unresolved_mask]["department"].value_counts().reset_index()
	unresolved_df.columns = ["department", "unresolved_count"]
	fig_unresolved = px.bar(
		unresolved_df,
		x="department",
		y="unresolved_count",
		title="Unresolved Complaints per Department",
		color="unresolved_count",
		color_continuous_scale="Sunsetdark",
		labels={"department": "Department", "unresolved_count": "Unresolved Complaints"},
		hover_data={"unresolved_count": ":.0f", "department": True},
	)
	_style_figure(fig_unresolved, x_title="Department", y_title="Unresolved Complaints")
	fig_unresolved.update_traces(
		hovertemplate="Department: %{x}<br>Unresolved: %{y}<extra></extra>"
	)
	st.plotly_chart(fig_unresolved, width="stretch")


def _render_trends_tab(df: pd.DataFrame, trend_df: pd.DataFrame, category_distribution_df: pd.DataFrame) -> None:
	"""Render trend and category distribution charts."""
	st.subheader("📈 Complaint Trends")

	if not trend_df.empty:
		fig_trend = px.line(
			trend_df,
			x="date",
			y="count",
			color="area",
			title="Daily Complaints by Area",
			markers=True,
			labels={"date": "Date", "count": "Complaint Count", "area": "Area"},
			hover_data={"count": ":.0f", "area": True},
		)
		_style_figure(fig_trend, x_title="Date", y_title="Complaint Count", legend_title="Area")
		fig_trend.update_traces(
			hovertemplate="Date: %{x|%Y-%m-%d}<br>Area: %{fullData.name}<br>Count: %{y}<extra></extra>"
		)
		st.plotly_chart(fig_trend, width="stretch")
	else:
		st.info("Trend data not available for current filters.")

	category_melt_df = category_distribution_df.melt(
		id_vars="area", var_name="category", value_name="count"
	)
	fig_stack = px.bar(
		category_melt_df,
		x="area",
		y="count",
		color="category",
		title="Category Distribution by Area",
		labels={"area": "Area", "count": "Complaint Count", "category": "Category"},
		hover_data={"count": ":.0f", "category": True},
	)
	fig_stack.update_layout(barmode="stack")
	_style_figure(fig_stack, x_title="Area", y_title="Complaint Count", legend_title="Category")
	fig_stack.update_traces(
		hovertemplate="Area: %{x}<br>Category: %{fullData.name}<br>Count: %{y}<extra></extra>"
	)
	st.plotly_chart(fig_stack, width="stretch")

	st.markdown("### 📊 Growth Rate & Smart Insights")

	trend_base = df.copy()
	trend_base["date"] = pd.to_datetime(trend_base["date"], errors="coerce")
	trend_base = trend_base.dropna(subset=["date"])

	if trend_base.empty:
		st.info("Insufficient date data to compute growth trends.")
		return

	max_date = trend_base["date"].max().normalize()
	current_start = max_date - pd.Timedelta(days=6)
	previous_start = current_start - pd.Timedelta(days=7)
	previous_end = current_start - pd.Timedelta(days=1)

	current_week = trend_base[(trend_base["date"] >= current_start) & (trend_base["date"] <= max_date)]
	previous_week = trend_base[
		(trend_base["date"] >= previous_start) & (trend_base["date"] <= previous_end)
	]

	curr_counts = current_week.groupby("area").size().rename("current_count")
	prev_counts = previous_week.groupby("area").size().rename("previous_count")
	growth_df = (
		pd.concat([curr_counts, prev_counts], axis=1)
		.fillna(0)
		.reset_index()
		.rename(columns={"index": "area"})
	)
	growth_df["current_count"] = growth_df["current_count"].astype(int)
	growth_df["previous_count"] = growth_df["previous_count"].astype(int)
	growth_df["growth_rate_pct"] = growth_df.apply(
		lambda r: ((r["current_count"] - r["previous_count"]) / r["previous_count"] * 100)
		if r["previous_count"] > 0
		else (100.0 if r["current_count"] > 0 else 0.0),
		axis=1,
	)
	growth_df = growth_df.sort_values("growth_rate_pct", ascending=False)

	fig_growth = px.bar(
		growth_df,
		x="area",
		y="growth_rate_pct",
		color="growth_rate_pct",
		color_continuous_scale="RdYlGn_r",
		title="Weekly Complaint Growth Rate by Area",
		labels={"area": "Area", "growth_rate_pct": "Growth Rate (%)"},
		hover_data={
			"current_count": True,
			"previous_count": True,
			"growth_rate_pct": ":.1f",
		},
	)
	_style_figure(fig_growth, x_title="Area", y_title="Growth Rate (%)")
	fig_growth.update_traces(
		hovertemplate=(
			"Area: %{x}<br>Growth: %{y:.1f}%<br>Current Week: %{customdata[0]}"
			"<br>Previous Week: %{customdata[1]}<extra></extra>"
		)
	)
	st.plotly_chart(fig_growth, width="stretch")

	insights: list[str] = []
	positive_growth = growth_df[growth_df["growth_rate_pct"] > 0].head(2)
	for _, row in positive_growth.iterrows():
		insights.append(
			f"{row['area']} shows a {row['growth_rate_pct']:.1f}% increase in complaints this week "
			f"({int(row['previous_count'])} → {int(row['current_count'])})."
		)

	high_volume = growth_df.sort_values("current_count", ascending=False).head(1)
	if not high_volume.empty:
		r = high_volume.iloc[0]
		insights.append(
			f"{r['area']} has the highest current weekly complaint load with {int(r['current_count'])} cases."
		)

	insights = insights[:3]
	if insights:
		for text in insights:
			st.info(f"💡 {text}")
	else:
		st.info("💡 No significant week-over-week growth detected across selected filters.")


def _render_action_brief(action_brief_df: pd.DataFrame) -> None:
	"""Render top-3 risk action brief cards."""
	st.markdown("---")
	st.subheader("🔥 Today's Action Brief")

	if action_brief_df.empty:
		st.info("No action brief available for current filters.")
		return

	for _, row in action_brief_df.iterrows():
		area_name = str(row["area"])
		complaint_count = int(row["complaint_count"])
		avg_risk_pct = float(row["avg_escalation_prob"]) * 100
		recommendation = f"Immediate intervention required in {area_name}."

		st.error(
			"🚨 High-Risk Area Alert\n\n"
			f"Area: {area_name}\n"
			f"Total complaints: {complaint_count}\n"
			f"Average escalation probability: {avg_risk_pct:.1f}%\n\n"
			f"Recommendation: {recommendation}"
		)


def _generate_ai_insights(df: pd.DataFrame) -> list[str]:
	"""Generate concise, impact-focused operational insights from complaint data."""
	insights: list[str] = []
	if df.empty:
		return insights

	# 1) Area with highest complaints
	area_counts = df["area"].value_counts()
	if not area_counts.empty:
		top_area = str(area_counts.index[0])
		top_area_count = int(area_counts.iloc[0])
		insights.append(f"📍 {top_area} has the highest complaint volume ({top_area_count} cases).")

	# 2) Area with highest escalation risk
	if "escalation_prob" in df.columns:
		risk_by_area = (
			df.groupby("area", as_index=False)["escalation_prob"]
			.mean()
			.sort_values("escalation_prob", ascending=False)
		)
		if not risk_by_area.empty:
			risk_area = str(risk_by_area.iloc[0]["area"])
			risk_score = float(risk_by_area.iloc[0]["escalation_prob"]) * 100
			insights.append(
				f"⚠️ {risk_area} shows the highest escalation risk ({risk_score:.1f}% average)."
			)

	# 3) Category with most unresolved issues
	unresolved_mask = _resolve_unresolved_mask(df)
	unresolved_df = df[unresolved_mask]
	if not unresolved_df.empty:
		category_counts = unresolved_df["category"].value_counts()
		if not category_counts.empty:
			top_category = str(category_counts.index[0])
			top_category_count = int(category_counts.iloc[0])
			insights.append(
				f"🧩 {top_category.title()} issues lead unresolved workload ({top_category_count} open complaints)."
			)

	return insights[:3]


def _render_ai_insights(df: pd.DataFrame) -> None:
	"""Render AI insights section with concise actionable highlights."""
	st.markdown("---")
	st.subheader("🤖 AI Insights")
	insights = _generate_ai_insights(df)

	if not insights:
		st.info("No insights available for the current filter selection.")
		return

	for item in insights:
		st.info(item)


def _render_predicted_risk_areas(df: pd.DataFrame) -> None:
	"""Render simple 7-day forecast of high-risk areas."""
	st.markdown("---")
	st.subheader("Predicted High Risk Areas in Next 7 Days")

	forecast_df = get_predicted_high_risk_areas(df, top_n=3)
	if forecast_df.empty:
		st.info("No forecast available for the current filter selection.")
		return

	for _, row in forecast_df.iterrows():
		st.warning(
			f"{row['area']}: score {row['prediction_score']:.1f} | "
			f"trend {row['growth_rate_pct']:.1f}% | "
			f"unresolved {row['unresolved_ratio'] * 100:.1f}%"
		)


def _generate_recommendations(df: pd.DataFrame) -> list[str]:
	"""Generate actionable recommendations from complaint load and escalation risk."""
	if df.empty or "escalation_prob" not in df.columns:
		return []

	unresolved_mask = _resolve_unresolved_mask(df)
	area_summary = (
		df.groupby("area", as_index=False)
		.agg(
			complaint_count=("area", "size"),
			avg_escalation_prob=("escalation_prob", "mean"),
		)
		.sort_values(["complaint_count", "avg_escalation_prob"], ascending=[False, False])
	)
	if area_summary.empty:
		return []

	max_count = max(float(area_summary["complaint_count"].max()), 1.0)
	area_summary["priority_score"] = (
		(area_summary["complaint_count"] / max_count) * 0.5
		+ area_summary["avg_escalation_prob"].clip(lower=0, upper=1) * 0.5
	)
	top_areas = area_summary.sort_values("priority_score", ascending=False).head(3)

	action_map = {
		"road": "deploy rapid road maintenance crews and temporary traffic safety control.",
		"waste": "increase waste collection frequency and place emergency cleanup teams.",
		"water": "dispatch WASA response units for leak/pressure diagnostics and immediate repair.",
		"traffic": "coordinate Traffic Police for signal optimization and congestion rerouting.",
	}

	recommendations: list[str] = []
	for _, row in top_areas.iterrows():
		area_name = str(row["area"])
		complaint_count = int(row["complaint_count"])
		escalation_pct = float(row["avg_escalation_prob"]) * 100

		area_unresolved = df[(df["area"].astype(str) == area_name) & unresolved_mask]
		if area_unresolved.empty:
			area_unresolved = df[df["area"].astype(str) == area_name]

		dominant_category = (
			str(area_unresolved["category"].mode().iloc[0]) if not area_unresolved.empty else "road"
		)
		action = action_map.get(dominant_category, "activate multi-agency rapid response coordination.")

		recommendations.append(
			f"{area_name}: {complaint_count} complaints with {escalation_pct:.1f}% average escalation risk; "
			f"recommended action: {action}"
		)

	return recommendations


def _render_recommendation_engine(df: pd.DataFrame) -> None:
	"""Render actionable authority recommendations."""
	st.markdown("---")
	st.subheader("🧭 Recommendation Engine")

	recommendations = _generate_recommendations(df)
	if not recommendations:
		st.info("No recommendations available for the current filter selection.")
		return

	for item in recommendations:
		st.warning(f"📌 {item}")


def _render_tabs(
	full_df: pd.DataFrame,
	predicted_df: pd.DataFrame,
	hotspot_df: pd.DataFrame,
	trend_df: pd.DataFrame,
	category_distribution_df: pd.DataFrame,
	model: RandomForestClassifier,
	feature_names: list[str],
) -> None:
	"""Render role-based tabs and isolate admin functionality."""
	is_admin = st.session_state.get("role") == "admin"

	if is_admin:
		dashboard_tab, hotspots_tab, trends_tab, admin_tab = st.tabs(
			["📊 Dashboard", "🔥 Hotspots", "📈 Trends", "🛠 Admin Panel"]
		)
	else:
		dashboard_tab, hotspots_tab, trends_tab = st.tabs(
			["📊 Dashboard", "🔥 Hotspots", "📈 Trends"]
		)

	with dashboard_tab:
		st.markdown("### ⚠️ Escalation Intelligence")
		_render_escalation_tab(predicted_df, model, feature_names)
		st.markdown("---")
		_render_department_tab(predicted_df)

	with hotspots_tab:
		_render_hotspot_tab(predicted_df, hotspot_df)

	with trends_tab:
		_render_trends_tab(predicted_df, trend_df, category_distribution_df)

	if is_admin:
		with admin_tab:
			_render_admin_panel(full_df)


def _render_admin_panel(df: pd.DataFrame) -> None:
	"""Render admin controls for full dataset view and complaint resolution updates."""
	if st.session_state.get("role") != "admin":
		return

	st.subheader("🛠 Admin Panel")
	st.markdown("### Admin Control Dashboard")
	st.caption("Filter complaints, identify unresolved cases, and resolve them in one click.")

	admin_df = df.copy()
	if "resolved" not in admin_df.columns:
		admin_df["resolved"] = 0

	admin_df["resolved"] = pd.to_numeric(admin_df["resolved"], errors="coerce").fillna(0).astype(int)
	admin_df["status"] = admin_df["resolved"].map({1: "resolved", 0: "unresolved"})
	admin_df["resolution_state"] = admin_df["resolved"].map({1: "🟢 Resolved", 0: "🔴 Unresolved"})

	filter_col_1, filter_col_2, filter_col_3 = st.columns([1.2, 1.2, 1])
	all_areas = sorted(admin_df["area"].astype(str).unique().tolist()) if "area" in admin_df.columns else []
	all_categories = (
		sorted(admin_df["category"].astype(str).unique().tolist()) if "category" in admin_df.columns else []
	)

	selected_areas = filter_col_1.multiselect(
		"Filter by Area",
		options=all_areas,
		default=all_areas,
		key="admin_filter_areas",
	)
	selected_categories = filter_col_2.multiselect(
		"Filter by Category",
		options=all_categories,
		default=all_categories,
		key="admin_filter_categories",
	)
	status_filter = filter_col_3.selectbox(
		"Filter by Status",
		options=["all", "unresolved", "resolved"],
		index=0,
		key="admin_filter_status",
	)

	search_text = st.text_input(
		"Search by Complaint ID or Text",
		placeholder="Type complaint ID or keywords...",
		key="admin_filter_search",
	)

	filtered_df = admin_df.copy()
	if selected_areas:
		filtered_df = filtered_df[filtered_df["area"].astype(str).isin(selected_areas)]
	if selected_categories:
		filtered_df = filtered_df[filtered_df["category"].astype(str).isin(selected_categories)]
	if status_filter != "all":
		filtered_df = filtered_df[filtered_df["status"].astype(str).str.lower().eq(status_filter)]
	if search_text.strip():
		search_value = search_text.strip().lower()
		id_match = filtered_df["complaint_id"].astype(str).str.lower().str.contains(search_value, na=False)
		text_col = "complaint_text" if "complaint_text" in filtered_df.columns else "text"
		text_match = filtered_df[text_col].astype(str).str.lower().str.contains(search_value, na=False)
		filtered_df = filtered_df[id_match | text_match]

	filtered_total = int(len(filtered_df))
	filtered_unresolved = int(filtered_df["resolved"].eq(0).sum()) if "resolved" in filtered_df.columns else 0
	filtered_resolved = int(filtered_total - filtered_unresolved)

	badge_col_1, badge_col_2, badge_col_3 = st.columns(3)
	badge_col_1.metric("Filtered Complaints", filtered_total)
	badge_col_2.metric("🔴 Unresolved", filtered_unresolved)
	badge_col_3.metric("🟢 Resolved", filtered_resolved)

	table_columns = [
		"complaint_id",
		"date",
		"area",
		"category",
		"priority",
		"days_to_resolve",
		"resolution_state",
		"complaint_text",
	]
	table_columns = [col for col in table_columns if col in filtered_df.columns]
	view_df = filtered_df[table_columns].copy()

	def _highlight_unresolved_rows(row: pd.Series) -> list[str]:
		if str(row.get("resolution_state", "")).startswith("🔴"):
			return ["background-color: rgba(215, 38, 61, 0.16);"] * len(row)
		return [""] * len(row)

	st.dataframe(
		view_df.style.apply(_highlight_unresolved_rows, axis=1),
		width="stretch",
		hide_index=True,
	)

	action_df = filtered_df[filtered_df["resolved"].eq(0)].copy()
	if action_df.empty:
		st.success("No unresolved complaints in the current filtered view.")
		return

	action_columns = [
		"complaint_id",
		"date",
		"area",
		"category",
		"priority",
		"days_to_resolve",
		"complaint_text",
	]
	action_columns = [col for col in action_columns if col in action_df.columns]
	action_editor_df = action_df[action_columns].copy()
	action_editor_df["mark_resolved"] = False

	st.markdown("#### Mark as Resolved")
	edited_action_df = st.data_editor(
		action_editor_df,
		width="stretch",
		hide_index=True,
		disabled=[col for col in action_editor_df.columns if col != "mark_resolved"],
		column_config={
			"mark_resolved": st.column_config.CheckboxColumn(
				"Mark as Resolved",
				help="Tick one or more unresolved complaints.",
				default=False,
			)
		},
		key="admin_mark_resolved_editor",
	)

	action_col_1, action_col_2 = st.columns(2)
	mark_selected = action_col_1.button("✅ Apply Selected", use_container_width=True)
	mark_all_filtered = action_col_2.button(
		"⚡ Resolve All Filtered", use_container_width=True
	)

	if mark_selected or mark_all_filtered:
		if mark_all_filtered:
			ids_to_resolve = action_editor_df["complaint_id"].astype(str).tolist()
		else:
			ids_to_resolve = (
				edited_action_df.loc[edited_action_df["mark_resolved"].astype(bool), "complaint_id"]
				.astype(str)
				.tolist()
			)
		if not ids_to_resolve:
			st.warning("No complaints selected for resolution.")
			return

		master_df = st.session_state.get("master_df", df).copy()
		if "resolved" not in master_df.columns:
			master_df["resolved"] = 0
		if "status" not in master_df.columns:
			master_df["status"] = "unresolved"

		selection_mask = master_df["complaint_id"].astype(str).isin(ids_to_resolve)
		master_df.loc[selection_mask, "resolved"] = 1
		master_df.loc[selection_mask, "status"] = "resolved"

		st.session_state["master_df"] = _optimize_dataframe_memory(master_df)
		_save_dataset(st.session_state["master_df"])
		st.success(f"Updated {int(selection_mask.sum())} complaints as resolved.")
		if hasattr(st, "experimental_rerun"):
			st.experimental_rerun()
		st.rerun()


def main() -> None:
	"""Application entrypoint for CivicMind dashboard."""
	_inject_custom_styles()
	_init_session_state()

	if not st.session_state.get("logged_in", False):
		_render_login()
		return

	st.title(APP_TITLE)
	_render_user_header()

	base_df = st.session_state.get("master_df", load_data())
	if base_df.empty:
		st.error("No complaint data available.")
		return

	try:
		model, encoders, feature_names = train_escalation_pipeline(base_df)
	except Exception as ex:
		st.error(f"Model training failed: {ex}")
		return

	render_submit_complaint(base_df, model, encoders)
	render_submission_result()

	df = _normalize_dates(_build_current_dataset(base_df))

	filtered_df = apply_filters(df)
	if filtered_df.empty:
		st.warning("No data available for the selected filters.")
		return

	try:
		pred_df = predict_escalation(filtered_df, model, encoders)
	except Exception as ex:
		st.error(f"Escalation prediction failed: {ex}")
		return

	pred_df = _optimize_dataframe_memory(pred_df)
	hotspot_df, trend_df, category_distribution_df, action_brief_df = _compute_cached_views(pred_df)
	_render_metrics(pred_df)
	_render_tabs(df, pred_df, hotspot_df, trend_df, category_distribution_df, model, feature_names)
	_render_ai_insights(pred_df)
	_render_predicted_risk_areas(pred_df)
	_render_recommendation_engine(pred_df)
	_render_action_brief(action_brief_df)


if __name__ == "__main__":
	main()
