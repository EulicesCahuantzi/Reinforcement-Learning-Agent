# Reinforcement-Learning-Agent
CartPole-v1 RL agent using tabular Q-learning with state discretization (Gymnasium), with training curves and evaluation metrics.
# CartPole Q-Learning Agent (Gymnasium)

A Python reinforcement learning project where an agent learns to balance a pole on a cart in the **CartPole-v1** simulation environment using **tabular Q-learning** with **state discretization**.

---

## Project Highlights

- **Environment:** Gymnasium `CartPole-v1`
- **Algorithm:** Tabular **Q-learning** (TD control)
- **Exploration:** **ε-greedy** with decay
- **State Handling:** Continuous observation → discretized bins → single integer state ID
- **Artifacts Saved:** Learning curve plot, Q-table, run summary JSON

---

## Repo Structure

rl_cartpole_qlearning_project/
├─ rl_cartpole_qlearning.py # Train + evaluate agent
├─ requirements.txt # Dependencies
├─ outputs/ # Generated artifacts after running
│ ├─ learning_curve.png
│ ├─ q_table.npy
│ └─ run_summary.json
└─ report/
└─ RL_CartPole_Qlearning_Report.docx


---

## Requirements

- **Python:** 3.10–3.12 recommended  
  *(Some RL libraries may have limited support on 3.13.)*

---

## Setup

### Clone and enter the project folder
1) Install Python 3.10–3.12.
2) Create a virtual environment and run: `pip install -r requirements.txt`.
3) Train: `python rl_cartpole_qlearning.py --episodes 1500`.
4) Check `outputs/learning_curve.png` and `outputs/run_summary.json`.clea

