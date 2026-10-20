"""Tabular Q-learning on Taxi-v3 (baseline for the DQN agent).

The Q-table has one row per state (500) and one column per action (6).
With the default ``--alpha 1.0`` the update reduces to the rule used in the
report, Q(s, a) <- r + gamma * max_a' Q(s', a'), which is exact here because
Taxi is deterministic.

    python src/q_learning_taxi.py train     # train from scratch -> runs/
    python src/q_learning_taxi.py test      # evaluate the pretrained Q-table
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from common import MODELS_DIR, RUNS_DIR, plot_rewards, set_seed, summarize

ENV_ID = "Taxi-v3"
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.99997  # applied after every environment step


def train(args):
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = gym.make(ENV_ID)
    env.action_space.seed(args.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = EPS_START

    rewards = []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        episode_reward, done = 0.0, False
        while not done:
            if rng.random() < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = int(np.argmax(q_table[state]))  # exploit
            epsilon = max(EPS_END, epsilon * EPS_DECAY)

            next_state, reward, terminated, truncated, _ = env.step(action)
            target = reward + GAMMA * np.max(q_table[next_state]) * (not terminated)
            q_table[state, action] += args.alpha * (target - q_table[state, action])

            state = next_state
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)

        if (episode + 1) % args.log_every == 0:
            print(
                f"episode {episode + 1:6d}  "
                f"mean reward (last {args.log_every}): {np.mean(rewards[-args.log_every:]):8.2f}  "
                f"epsilon: {epsilon:.3f}"
            )
    env.close()

    args.model.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.model, q_table, delimiter=",")
    print(f"Saved Q-table to {args.model}")
    plot_rewards(rewards, 50, "Taxi - Q-learning training", RUNS_DIR / "taxi-qlearning-training.png")


def test(args):
    set_seed(args.seed)
    env = gym.make(ENV_ID)
    q_table = np.loadtxt(args.model, delimiter=",")
    print(f"Loaded Q-table from {args.model}")

    rewards = []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        episode_reward, done = 0.0, False
        while not done:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q_table[state])))
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    env.close()

    summarize(rewards)
    plot_rewards(rewards, 20, "Taxi - Q-learning testing", RUNS_DIR / "taxi-qlearning-testing.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--episodes", type=int, help="default: 10000 for train, 1000 for test")
    parser.add_argument("--alpha", type=float, default=1.0, help="learning rate of the Q-update")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, help="Q-table CSV to write (train) or read (test)")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    if args.mode == "train":
        args.episodes = args.episodes or 10_000
        args.model = args.model or RUNS_DIR / "taxi_q_table.csv"
        train(args)
    else:
        args.episodes = args.episodes or 1_000
        args.model = args.model or MODELS_DIR / "taxi_q_table.csv"
        test(args)


if __name__ == "__main__":
    main()
