# MediAI: Intelligent Medical Emergency Agent

**CS24308 Artificial Intelligence Lab | Session MO-26 | B.Tech CSE**

Team: Abhijeet Gorai (BTECH/10024/24), Nilesh Kumar (BTECH/10054/24), Yash Raj (BTECH/10047/24)

Live app: PASTE YOUR STREAMLIT LINK HERE

## What it does
MediAI is a single AI agent that **perceives** a patient's symptoms, **reasons** about them, and **acts**:
it triages the patient, estimates diabetes risk, dispatches an ambulance along the fastest route, and builds a rescue plan.

| Syllabus module | Technique | Where in the project |
|---|---|---|
| I. Preliminaries | Intelligent agent (perceive, reason, act) | Agent mission tab |
| II. Search | BFS, Greedy, A* | Ambulance routing tab |
| III. Knowledge and reasoning | Rule base + forward chaining | Triage (RED/ORANGE/YELLOW/GREEN) |
| IV. Planning and probabilistic reasoning | Goal-stack planning, Bayes' theorem | Rescue plan, Bayes tab |
| V. Learning | Neural network (MLP 64-32) | Risk checker, AUC 0.824 |

## Dataset
CDC Diabetes Health Indicators (UCI repository id 891), 253,680 patients, 21 features.

## Files
- `app.py`: Streamlit web app
- `agent.py`: search, triage rules, Bayes, planner, mission runner
- `mediai_model.joblib`: trained neural network, scaler, feature list
- `requirements.txt`: dependencies (Python 3.12 on Streamlit Cloud)

Educational project. Not medical advice.
