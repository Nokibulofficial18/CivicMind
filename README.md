# 🏙️ CivicMind AI — Dhaka Urban Intelligence Dashboard

CivicMind AI is a role-based, data-driven civic analytics platform built for hackathon demonstration and smart city operations. It helps authorities in Dhaka detect complaint hotspots, forecast risk, and take proactive action.

---

## 🌍 Project Overview

Urban complaint handling is usually reactive and fragmented across departments. CivicMind AI unifies:

- complaint intake,
- hotspot intelligence,
- escalation prediction,
- admin resolution workflow,
- and actionable recommendations

into one interactive dashboard.

---

## ❗ Problem Statement

Dhaka receives frequent complaints across road, waste, water, and traffic categories. Manual workflows struggle with:

- delayed identification of critical areas,
- high unresolved complaint backlogs,
- weak prioritization across departments,
- limited foresight into near-future risk.

---

## ✅ Solution

CivicMind AI provides a decision-support system where:

- **Users** submit complaints and monitor city-level intelligence.
- **Admins** filter complaints, resolve cases, and persist updates.
- The platform continuously computes:
  - hotspot severity,
  - escalation probability,
  - root-cause analysis,
  - predicted high-risk areas (next 7 days),
  - and recommendation actions for authorities.

---

## 🚀 Core Features

- 🔐 **Role-based Authentication** (User / Admin)
- 📝 **Submit Complaint** with instant escalation scoring
- 🔥 **Hotspot Ranking** using weighted severity score
- 🗺️ **Folium Risk Heatmap** (weighted by escalation probability)
- ⚠️ **Escalation Model** (RandomForestClassifier)
- 📈 **Trend + Growth Analytics**
- 🧭 **Root Cause Analysis** per hotspot area:
  - growth trend,
  - unresolved ratio,
  - dominant category
- 🔮 **Predicted High Risk Areas in Next 7 Days**
- 📌 **Recommendation Engine** for authority actions
- 🤖 **AI Insights** and **Today’s Action Brief**
- 🛠️ **Admin Panel Tab** with:
  - unresolved highlighting,
  - advanced filters,
  - per-row resolve actions,
  - CSV persistence

---

## 🧠 Tech Stack

- **App/UI:** Streamlit
- **Data:** Pandas, NumPy
- **Machine Learning:** scikit-learn
- **Visualizations:** Plotly, Folium
- **Storage:** CSV (`data/complaints.csv`)
- **Language:** Python

---

## 🗂️ Project Structure

```text
CivicMind AI/
├── app.py
├── data/
│   ├── generate_data.py
│   └── complaints.csv
├── models/
│   ├── escalation.py
│   └── hotspot.py
├── utils/
│   └── map_utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1) Clone

```bash
git clone <your-repo-url>
cd "cognisor ai"
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Launch

```bash
streamlit run app.py
```

If port `8501` is busy:

```bash
streamlit run app.py --server.port 8502
```

---

## 🔐 Demo Credentials

```text
Normal User
username: user1
password: 1234

Admin
username: admin
password: admin123
```

---



## 🏆 Hackathon Note

This project was built as a **hackathon prototype** focused on:

- smart city governance,
- predictive civic intelligence,
- and operational decision support for Dhaka.

---

## 📌 Future Enhancements

- secure credential storage + password hashing
- database backend (PostgreSQL)
- audit logs for admin actions
- API integration with live complaint portals
- cloud deployment + CI/CD

---

## 📄 License

For educational and hackathon demonstration use.
