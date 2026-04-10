from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


NUM_ROWS = 600


AREA_POPULATION = {
	"Mirpur": 1200000,
	"Dhanmondi": 350000,
	"Uttara": 600000,
	"Farmgate": 250000,
	"Demra": 500000,
	"Gulshan": 300000,
	"Mohammadpur": 700000,
}


# Category weights by area (must sum close to 1.0 per area)
CATEGORY_WEIGHTS = {
	"Mirpur": {"road": 0.36, "waste": 0.30, "water": 0.17, "traffic": 0.17},
	"Dhanmondi": {"road": 0.22, "waste": 0.21, "water": 0.23, "traffic": 0.34},
	"Uttara": {"road": 0.24, "waste": 0.22, "water": 0.22, "traffic": 0.32},
	"Farmgate": {"road": 0.22, "waste": 0.20, "water": 0.18, "traffic": 0.40},
	"Demra": {"road": 0.30, "waste": 0.27, "water": 0.28, "traffic": 0.15},
	"Gulshan": {"road": 0.18, "waste": 0.17, "water": 0.25, "traffic": 0.40},
	"Mohammadpur": {"road": 0.34, "waste": 0.31, "water": 0.19, "traffic": 0.16},
}


TEXT_TEMPLATES = {
	"road": [
		"Large pothole near the main road in {area} causing bike accidents.",
		"Broken pavement and uneven road surface reported in {area}.",
		"Drain cover is missing on a busy road in {area}.",
		"Road repair work is incomplete in {area}, creating traffic bottlenecks.",
		"Street section in {area} is flooded after rain due to damaged road drainage.",
	],
	"waste": [
		"Garbage has not been collected for several days in {area}.",
		"Overflowing community bin in {area} is causing foul smell.",
		"Illegal dumping spotted beside a residential lane in {area}.",
		"Waste pile near market area in {area} attracting stray animals.",
		"Blocked drain in {area} due to accumulated plastic waste.",
	],
	"water": [
		"Low water pressure reported in multiple households in {area}.",
		"Residents in {area} report muddy tap water during morning hours.",
		"Intermittent water supply disruption observed in {area}.",
		"Leakage from underground supply line in {area} wasting clean water.",
		"No municipal water available in part of {area} since last night.",
	],
	"traffic": [
		"Severe traffic congestion at peak hour in {area} intersection.",
		"Signal light malfunction causing long queues in {area}.",
		"Unauthorized roadside parking worsening traffic flow in {area}.",
		"Frequent gridlock near bus stop reported in {area}.",
		"Traffic police presence is low in {area} during rush hours.",
	],
}


CATEGORY_PRIORITY_SCORE = {
	"road": 0.55,
	"waste": 0.45,
	"water": 0.50,
	"traffic": 0.60,
}


def weighted_choice(rng: np.random.Generator, options: dict[str, float]) -> str:
	keys = list(options.keys())
	probs = np.array(list(options.values()), dtype=float)
	probs = probs / probs.sum()
	return str(rng.choice(keys, p=probs))


def choose_priority(
	rng: np.random.Generator,
	area: str,
	category: str,
	population_min: int,
	population_max: int,
) -> str:
	pop_score = (AREA_POPULATION[area] - population_min) / (population_max - population_min)
	base = CATEGORY_PRIORITY_SCORE[category]
	noise = rng.normal(0.0, 0.08)
	priority_score = 0.5 * base + 0.4 * pop_score + noise

	if priority_score >= 0.66:
		return "high"
	if priority_score >= 0.42:
		return "medium"
	return "low"


def generate_complaints(num_rows: int = NUM_ROWS, seed: int = 42) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	today = datetime.now().date()

	areas = list(AREA_POPULATION.keys())
	area_probs = np.array(list(AREA_POPULATION.values()), dtype=float)
	area_probs = area_probs / area_probs.sum()

	pop_min = min(AREA_POPULATION.values())
	pop_max = max(AREA_POPULATION.values())

	rows: list[dict[str, object]] = []

	for i in range(1, num_rows + 1):
		area = str(rng.choice(areas, p=area_probs))
		category = weighted_choice(rng, CATEGORY_WEIGHTS[area])

		age_days = int(rng.integers(0, 60))
		complaint_date = today - timedelta(days=age_days)

		unresolved_prob = min(0.85, 0.15 + (age_days / 59) * 0.65)
		status = "unresolved" if rng.random() < unresolved_prob else "resolved"

		priority = choose_priority(rng, area, category, pop_min, pop_max)

		if status == "unresolved":
			days_to_resolve = int(rng.integers(5, 21))
		else:
			days_to_resolve = int(rng.integers(1, 8))

		template = str(rng.choice(TEXT_TEMPLATES[category]))
		complaint_text = template.format(area=area)

		rows.append(
			{
				"complaint_id": f"CM-{i:04d}",
				"date": complaint_date.isoformat(),
				"area": area,
				"category": category,
				"priority": priority,
				"status": status,
				"days_to_resolve": days_to_resolve,
				"complaint_text": complaint_text,
			}
		)

	return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
	print("\n=== CivicMind Complaint Data Summary ===")
	print(f"Total rows: {len(df)}")
	print(f"Date range: {df['date'].min()} to {df['date'].max()}")

	print("\nComplaints by area:")
	print(df["area"].value_counts().sort_index().to_string())

	print("\nComplaints by category:")
	print(df["category"].value_counts().sort_index().to_string())

	print("\nStatus distribution:")
	print(df["status"].value_counts().to_string())

	print("\nPriority distribution:")
	print(df["priority"].value_counts().to_string())

	print("\nAverage days_to_resolve by status:")
	print(df.groupby("status")["days_to_resolve"].mean().round(2).to_string())


def main() -> None:
	df = generate_complaints(num_rows=NUM_ROWS, seed=42)

	output_path = Path(__file__).resolve().parent / "complaints.csv"
	df.to_csv(output_path, index=False)

	print(f"Saved dataset to: {output_path}")
	print_summary(df)


if __name__ == "__main__":
	main()
