from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap


# Real Dhaka area coordinates (latitude, longitude)
AREA_COORDINATES: dict[str, tuple[float, float]] = {
	"Mirpur": (23.8223, 90.3654),
	"Dhanmondi": (23.7465, 90.3760),
	"Uttara": (23.8759, 90.3795),
	"Farmgate": (23.7580, 90.3890),
	"Demra": (23.7114, 90.4794),
	"Gulshan": (23.7925, 90.4078),
	"Mohammadpur": (23.7641, 90.3587),
}


def generate_heatmap(df: pd.DataFrame) -> folium.Map:
	"""Create and save a weighted complaint heatmap for Dhaka.

	Expected columns in `df`: `area`, `escalation_prob`.

	For each row, area coordinates are looked up and a small random spatial jitter
	(±0.008 degrees) is applied to distribute points naturally.
	`escalation_prob` is used as heat intensity weight.

	The map is saved as `dhaka_heatmap.html` and returned.
	"""
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame.")
	if df.empty:
		raise ValueError("df is empty.")

	required_cols = {"area", "escalation_prob"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {sorted(missing)}")

	working = df.copy()
	working = working[working["area"].isin(AREA_COORDINATES)].copy()
	if working.empty:
		raise ValueError("No rows with valid Dhaka areas found in df['area'].")

	working["escalation_prob"] = pd.to_numeric(working["escalation_prob"], errors="coerce")
	working = working.dropna(subset=["escalation_prob"])
	if working.empty:
		raise ValueError("No valid numeric values found in 'escalation_prob'.")

	rng = np.random.default_rng(42)
	heat_data: list[list[float]] = []

	for _, row in working.iterrows():
		base_lat, base_lon = AREA_COORDINATES[str(row["area"])]
		lat = base_lat + float(rng.uniform(-0.005, 0.005))
		lon = base_lon + float(rng.uniform(-0.005, 0.005))
		weight = float(np.clip(row["escalation_prob"], 0.0, 1.0))
		heat_data.append([lat, lon, weight])

	map_obj = folium.Map(
		location=[23.8103, 90.4125],
		zoom_start=12,
		tiles="CartoDB positron",
		control_scale=True,
		prefer_canvas=True,
	)

	HeatMap(
		data=heat_data,
		radius=17,
		blur=18,
		max_zoom=13,
		min_opacity=0.35,
		gradient={
			0.00: "#2A9D8F",
			0.30: "#7BC96F",
			0.55: "#F4D35E",
			0.75: "#F18F01",
			1.00: "#D7263D",
		},
	).add_to(map_obj)

	title_html = """
	<div style="
		position: fixed;
		top: 14px;
		left: 50px;
		z-index: 9999;
		background: rgba(255,255,255,0.95);
		border: 1px solid #c9ced6;
		border-radius: 6px;
		padding: 8px 12px;
		font-size: 14px;
		font-weight: 700;
		color: #243447;
	">
		Dhaka Complaint Risk Heatmap
	</div>
	"""
	map_obj.get_root().html.add_child(folium.Element(title_html))

	legend_html = """
	<div style="
		position: fixed;
		bottom: 40px;
		left: 40px;
		z-index: 9999;
		background-color: rgba(255,255,255,0.95);
		border: 2px solid #666;
		border-radius: 6px;
		padding: 10px 12px 12px 12px;
		font-size: 13px;
		box-shadow: 0 1px 6px rgba(0,0,0,0.2);
	">
		<b>Escalation Risk (Heat Intensity)</b><br>
		<div style="
			margin: 6px 0 8px 0;
			height: 10px;
			width: 180px;
			border-radius: 4px;
			background: linear-gradient(to right, #2A9D8F, #F4D35E, #D7263D);
		"></div>
		<div style="display:flex; justify-content:space-between; width:180px; font-size:11px;">
			<span>Low</span><span>Medium</span><span>High</span>
		</div>
		<div style="margin-top:8px;"><b>Area Marker Color</b></div>
		<span style="color:#2A9D8F;">●</span> Stable hotspot<br>
		<span style="color:#F18F01;">●</span> Moderate hotspot<br>
		<span style="color:#D7263D;">●</span> Severe hotspot
	</div>
	"""
	map_obj.get_root().html.add_child(folium.Element(legend_html))

	output_path = Path("dhaka_heatmap.html")
	map_obj.save(str(output_path))

	return map_obj


def add_area_markers(map_obj: folium.Map, hotspot_df: pd.DataFrame) -> folium.Map:
	"""Add area-level hotspot circle markers on top of a Folium map.

	Expected columns in `hotspot_df`: `area`, `total`, `hotspot_score`.

	Marker color rule:
	- red: hotspot_score > 70
	- orange: hotspot_score > 40
	- green: otherwise
	"""
	if not isinstance(map_obj, folium.Map):
		raise TypeError("map_obj must be a folium.Map instance.")
	if not isinstance(hotspot_df, pd.DataFrame):
		raise TypeError("hotspot_df must be a pandas DataFrame.")
	if hotspot_df.empty:
		return map_obj

	required_cols = {"area", "total", "hotspot_score"}
	missing = required_cols - set(hotspot_df.columns)
	if missing:
		raise ValueError(f"Missing required columns in hotspot_df: {sorted(missing)}")

	for _, row in hotspot_df.iterrows():
		area = str(row["area"])
		if area not in AREA_COORDINATES:
			continue

		total = int(pd.to_numeric(row["total"], errors="coerce") if pd.notna(row["total"]) else 0)
		score = float(
			pd.to_numeric(row["hotspot_score"], errors="coerce")
			if pd.notna(row["hotspot_score"])
			else 0.0
		)

		if score > 70:
			color = "#D7263D"
		elif score > 40:
			color = "#F18F01"
		else:
			color = "#2A9D8F"

		lat, lon = AREA_COORDINATES[area]
		if total >= 120:
			radius = 13
		elif total >= 80:
			radius = 11
		elif total >= 40:
			radius = 9
		else:
			radius = 7

		severity = "Severe" if score > 70 else ("Moderate" if score > 40 else "Stable")
		popup_text = (
			"<div style='font-size:13px; line-height:1.35;'>"
			f"<b>{area}</b><br>"
			f"Total complaints: <b>{total}</b><br>"
			f"Hotspot score: <b>{score:.2f}</b><br>"
			f"Severity: <b>{severity}</b>"
			"</div>"
		)

		folium.CircleMarker(
			location=[lat, lon],
			radius=radius,
			color=color,
			fill=True,
			fill_color=color,
			fill_opacity=0.85,
			weight=2,
			popup=folium.Popup(popup_text, max_width=260),
			tooltip=f"{area} | Complaints: {total} | Score: {score:.1f}",
		).add_to(map_obj)

	return map_obj
