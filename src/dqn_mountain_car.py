"""Deep Q-learning on MountainCar-v0.

The observation is used as-is: [position, velocity].

    python src/dqn_mountain_car.py train    # train from scratch -> runs/
    python src/dqn_mountain_car.py test     # evaluate the pretrained model
    python src/dqn_mountain_car.py test --model runs/mountaincar_dqn.pth
"""

import argparse

import gymnasium as gym
import numpy as np

from common import MODELS_DIR, RUNS_DIR, plot_rewards, set_seed, summarize
from dqn import DQNAgent

ENV_ID = "MountainCar-v0"
STATE_DIM = 2
N_ACTIONS = 3

# Hyper-parameters used for the pretrained model
HPARAMS = dict(
    lr=1e-3,
    gamma=0.99,
    batch_size=128,
    memory_size=10_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.9997,  # applied once at the end of every episode
)


def train(args):
    set_seed(args.seed)
    env = gym.make(ENV_ID)
    env.action_space.seed(args.seed)
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=args.seed, **HPARAMS)

    rewards = []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        episode_reward, done = 0.0, False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, terminated)
            agent.learn()
            state = next_state
            episode_reward += reward
            done = terminated or truncated
        agent.decay_epsilon()
        rewards.append(episode_reward)

        if (episode + 1) % args.log_every == 0:
            print(
                f"episode {episode + 1:6d}  "
                f"mean reward (last {args.log_every}): {np.mean(rewards[-args.log_every:]):8.2f}  "
                f"epsilon: {agent.epsilon:.3f}"
            )
    env.close()

    agent.save(args.model)
    print(f"Saved model to {args.model}")
    plot_rewards(rewards, 500, "MountainCar - DQN training", RUNS_DIR / "mountaincar-dqn-training.png")


def test(args):
    set_seed(args.seed)
    env = gym.make(ENV_ID)
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=args.seed, **HPARAMS)
    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    rewards = []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        episode_reward, done = 0.0, False
        while not done:
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    env.close()

    summarize(rewards)
    plot_rewards(rewards, 20, "MountainCar - DQN testing", RUNS_DIR / "mountaincar-dqn-testing.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--episodes", type=int, help="default: 10000 for train, 1000 for test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", help="checkpoint to write (train) or read (test)")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "train":
        args.episodes = args.episodes or 10_000
        args.model = args.model or RUNS_DIR / "mountaincar_dqn.pth"
        train(args)
    else:
        args.episodes = args.episodes or 1_000
        args.model = args.model or MODELS_DIR / "mountaincar_dqn.pth"
        test(args)


if __name__ == "__main__":
    main()
