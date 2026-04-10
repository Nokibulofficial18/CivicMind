# 🏙️ CivicMind AI — Dhaka Urban Intelligence Dashboard

CivicMind AI is a data-driven urban governance platform built for hackathon use-cases. It helps city authorities monitor citizen complaints, detect operational hotspots, predict escalation risk, and prioritize action in real time.

---

## 🌍 Project Overview

Urban complaint handling is often reactive and fragmented. CivicMind AI brings complaint data, predictive intelligence, and operational workflows into one decision-support dashboard.

It is designed for teams such as:
- City Corporation
- WASA
- Traffic Police
- Urban control rooms and emergency planners

---

## ❗ Problem Statement

Dhaka receives high volumes of civic complaints across road damage, waste, water, and traffic. Traditional workflows face:
- Delayed response to high-risk complaints
- Limited visibility into area-wise hotspots
- Weak prioritization across departments
- No proactive escalation warning system

Result: unresolved complaints grow, citizen trust drops, and city operations become inefficient.

---

## ✅ Solution

CivicMind AI provides a role-based analytics dashboard that:
- Centralizes complaint intelligence
- Predicts complaint escalation probability
- Identifies high-risk areas and trends
- Supports operational decisions with AI-generated insights
- Enables admins to mark complaints resolved and instantly refresh analytics

---

## 🚀 Key Features

- 🔐 **Role-based authentication** (User / Admin)
- 📝 **Submit Complaint** workflow with instant risk analysis
- 🔥 **Hotspot detection** by area
- 🤖 **Escalation prediction** using Random Forest
- 🗺️ **Folium heatmap** with weighted risk visualization
- 📈 **Trend analytics** and growth-rate monitoring
- 🧩 **Department workload** analysis
- 💡 **AI Insights** auto-generated from live data
- 🚨 **Today’s Action Brief** for top high-risk areas
- 🛠️ **Admin Panel** for filtering and resolving complaints

---

## 🧠 Tech Stack

- **Frontend / App:** Streamlit
- **Data Processing:** Pandas, NumPy
- **ML:** scikit-learn (RandomForest, LabelEncoder)
- **Visualization:** Plotly, Folium
- **Persistence:** CSV-based dataset storage
- **Language:** Python 3.x

---

## 🗂️ Project Structure

```text
CivicMind AI/
├── app.py
├── data/
│   ├── generate_data.py
│   └── complaints.csv
├── models/
│   ├── hotspot.py
│   └── escalation.py
├── utils/
│   └── map_utils.py
└── requirements.txt
```

---

## ⚙️ How to Run

### 1) Clone the repository
```bash
git clone <your-repo-url>
cd "cognisor ai"
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Start the app
```bash
streamlit run app.py
```

### 4) Login credentials
```text
Normal User:
  username: user1
  password: 1234

Admin:
  username: admin
  password: admin123
```

---



## 🏆 Hackathon Context

This project was developed as a **hackathon prototype** to demonstrate how AI + analytics can improve public service delivery in Dhaka.

Focus areas:
- Smart city governance
- Predictive civic operations
- Actionable public-sector intelligence

---

## 📌 Future Improvements

- Database integration (PostgreSQL / Cloud)
- Secure password hashing + user management
- Real-time API ingestion from complaint portals
- NLP summarization for complaint narratives
- Deployment with CI/CD and monitoring

---

## 📄 License

This project is for educational and hackathon demonstration purposes.
