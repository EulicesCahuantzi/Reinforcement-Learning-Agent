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

### 1) Clone and enter the project folder

```bash
git clone <your-repo-url>
cd rl_cartpole_qlearning_project
2) Create a virtual environment + activate it
Windows (Command Prompt)

python -m venv .venv
.venv\Scripts\activate.bat
Windows (PowerShell)
If activation is blocked:

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
macOS / Linux

python -m venv .venv
source .venv/bin/activate
3) Install dependencies
pip install -r requirements.txt
Run the Agent
Train with default settings
python rl_cartpole_qlearning.py
Train with custom hyperparameters (example)
python rl_cartpole_qlearning.py --episodes 4000 --epsilon-decay 0.998 --alpha 0.10
Render evaluation episodes (optional)
Requires a working display. May not work in headless/remote setups.

python rl_cartpole_qlearning.py --render-eval
Outputs
After running, artifacts are saved to outputs/:

learning_curve.png — episode reward + moving average during training

q_table.npy — trained Q-table

run_summary.json — configuration + performance metrics

Example terminal output:

=== Training Complete ===
Solved at episode: None
Train avg reward:  177.01
Last-100 avg:      173.24
Eval avg reward:   144.25 ± 31.07
Solved at episode: None means this run did not reach the configured “solved” threshold.

How It Works
Environment
CartPole-v1 observations contain 4 continuous values:

cart position

cart velocity

pole angle

pole angular velocity

Actions are discrete:

0 = push left

1 = push right

Reward is typically +1 per timestep the pole remains balanced.

Discretization (Key Step)
Because Q-learning is tabular, the continuous observation is converted into a finite state space:

Each observation dimension is bucketed into bins

Bin indices are combined into a single state ID

Q-Learning Update Rule
𝑄
(
𝑠
,
𝑎
)
←
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max
⁡
𝑎
′
𝑄
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)←Q(s,a)+α[r+γ 
a 
′
 
max
​
 Q(s 
′
 ,a 
′
 )−Q(s,a)]
α = learning rate

γ = discount factor

ε-greedy exploration chooses random actions with probability ε

Notes / Limitations
Performance depends heavily on:

bin counts and ranges

exploration decay schedule

learning rate

Discretization is an approximation and can limit performance.

A strong next step is implementing DQN (Deep Q-Network) to learn directly from continuous states.

References (APA)
Farama Foundation. (n.d.). Gymnasium documentation. https://gymnasium.farama.org/

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279–292. https://doi.org/10.1007/BF00992698

::contentReference[oaicite:0]{index=0}
