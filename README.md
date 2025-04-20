# 🧠 Multiple Sclerosis Treatment Simulation

This project is a **Reinforcement Learning-based simulation** for treatment planning of **Multiple Sclerosis (MS)** using a custom Gym environment and Q-Learning.

Deployed using **Streamlit Cloud** 🚀

---

## 📋 Features

- Custom environment simulating MS symptoms: Symptom Severity, Inflammation, Fatigue
- RL agent learns to take actions like:
  - No Action
  - Medication
  - Lifestyle Change
  - Combo (Med + Lifestyle)
- Visualization using Streamlit

---

## 🛠️ How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/KSaiSrivalli/ms-rl-treatment-simulation.git
   cd ms-rl-treatment-simulation
   
📁 Project Structure
📦 ms-rl-treatment-simulation/
├── agent/
│   └── q_learning.py
├── environment/
│   └── ms_env.py
├── trainandsimulate.py
├── streamlitapp.py
├── q_table.pkl
├── requirements.txt
└── README.md

🧪 Sample Output
Reward: +7.5 — RL agent learns effective treatment over episodes

🌐 Deployment
Live on Streamlit Cloud  "https://ms-rl-treatment-simulation-ksaisrivalli.streamlit.app/"

📌 Author
K Sai Srivalli
Final Year B.Tech | AI in Healthcare Research
