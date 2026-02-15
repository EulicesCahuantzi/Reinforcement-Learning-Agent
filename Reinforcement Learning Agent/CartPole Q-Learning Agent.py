"""CartPole Q-Learning Agent
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


@dataclass
class RunConfig:
    env_id: str = "CartPole-v1"
    episodes: int = 1500
    max_steps: int = 500
    gamma: float = 0.99
    alpha: float = 0.15
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    bins_cart_pos: int = 6
    bins_cart_vel: int = 6
    bins_pole_angle: int = 12
    bins_pole_ang_vel: int = 12
    seed: int = 42
    moving_avg_window: int = 50
    solved_avg_reward: float = 475.0
    solved_window: int = 100


def make_bins(cfg: RunConfig) -> List[np.ndarray]:
    cart_pos = np.linspace(-4.8, 4.8, cfg.bins_cart_pos - 1)
    cart_vel = np.linspace(-3.0, 3.0, cfg.bins_cart_vel - 1)
    pole_angle = np.linspace(-0.418, 0.418, cfg.bins_pole_angle - 1)
    pole_ang_vel = np.linspace(-3.5, 3.5, cfg.bins_pole_ang_vel - 1)
    return [cart_pos, cart_vel, pole_angle, pole_ang_vel]


def discretize(obs: np.ndarray, bins: List[np.ndarray]) -> Tuple[int, int, int, int]:
    idxs = []
    for x, edges in zip(obs, bins):
        idxs.append(int(np.digitize(x, edges)))
    return tuple(idxs)  # type: ignore


def flatten_state(idxs: Tuple[int, int, int, int], dims: Tuple[int, int, int, int]) -> int:
    return int(np.ravel_multi_index(idxs, dims))


def epsilon_greedy(q: np.ndarray, s: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q.shape[1]))
    return int(np.argmax(q[s]))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if len(x) < window:
        return np.full_like(x, np.mean(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curve(rewards: np.ndarray, ma: np.ndarray, outpath: Path, window: int) -> None:
    plt.figure()
    plt.plot(rewards, label="Episode reward")
    if len(ma) > 1:
        x_ma = np.arange(window - 1, window - 1 + len(ma))
        plt.plot(x_ma, ma, label=f"Moving avg (window={window})")
    plt.title("CartPole Q-Learning: Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def evaluate_policy(
    cfg: RunConfig,
    q: np.ndarray,
    bins: List[np.ndarray],
    dims: Tuple[int, int, int, int],
    episodes: int = 20,
    render: bool = False,
) -> List[float]:
    env = gym.make(cfg.env_id, render_mode="human" if render else None)
    rewards: List[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + cfg.seed + ep)
        s = flatten_state(discretize(obs, bins), dims)
        total = 0.0
        for _ in range(cfg.max_steps):
            a = int(np.argmax(q[s]))
            next_obs, reward, terminated, truncated, _ = env.step(a)
            s = flatten_state(discretize(next_obs, bins), dims)
            total += float(reward)
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards


def train(cfg: RunConfig, render_eval: bool = False) -> dict:
    env = gym.make(cfg.env_id)
    rng = np.random.default_rng(cfg.seed)
    bins = make_bins(cfg)
    dims = (cfg.bins_cart_pos, cfg.bins_cart_vel, cfg.bins_pole_angle, cfg.bins_pole_ang_vel)
    n_states = int(np.prod(dims))
    n_actions = int(env.action_space.n)
    q = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = cfg.epsilon_start
    episode_rewards: List[float] = []
    solved_at = None

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + ep)
        s = flatten_state(discretize(obs, bins), dims)
        total_reward = 0.0

        for _ in range(cfg.max_steps):
            a = epsilon_greedy(q, s, epsilon, rng)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s2 = flatten_state(discretize(next_obs, bins), dims)

            target = reward + (0.0 if done else cfg.gamma * float(np.max(q[s2])))
            q[s, a] += cfg.alpha * (target - q[s, a])

            s = s2
            total_reward += float(reward)
            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        if len(episode_rewards) >= cfg.solved_window and solved_at is None:
            if float(np.mean(episode_rewards[-cfg.solved_window:])) >= cfg.solved_avg_reward:
                solved_at = int(ep) 

    env.close()

    rewards = np.array(episode_rewards, dtype=np.float32)
    ma = moving_average(rewards, cfg.moving_avg_window)

    plot_path = OUTPUTS_DIR / "learning_curve.png"
    plot_learning_curve(rewards, ma, plot_path, cfg.moving_avg_window)

    q_path = OUTPUTS_DIR / "q_table.npy"
    np.save(q_path, q)

    eval_rewards = evaluate_policy(cfg, q, bins, dims, episodes=20, render=render_eval)

    summary = {
        "config": asdict(cfg),
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "solved_at_episode": solved_at,
        "train_reward_mean": float(np.mean(rewards)),
        "train_reward_last_100_avg": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)),
        "eval_reward_mean": float(np.mean(eval_rewards)),
        "eval_reward_std": float(np.std(eval_rewards)),
        "artifacts": {
            "learning_curve_png": str(plot_path),
            "q_table_npy": str(q_path),
        },
    }

    summary_path = OUTPUTS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    summary["artifacts"]["run_summary_json"] = str(summary_path)

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Q-learning agent on CartPole-v1 (Gymnasium).")
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--render-eval", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    summary = train(cfg, render_eval=args.render_eval)

    print("\n=== Training Complete ===")
    print(f"Solved at episode: {summary['solved_at_episode']}")
    print(f"Train avg reward:  {summary['train_reward_mean']:.2f}")
    print(f"Last-100 avg:      {summary['train_reward_last_100_avg']:.2f}")
    print(f"Eval avg reward:   {summary['eval_reward_mean']:.2f} ± {summary['eval_reward_std']:.2f}")
    print("\nArtifacts:")
    for k, v in summary["artifacts"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
