"""Deep Q-learning on Taxi-v3.

The discrete Taxi state (an integer in 0..499) is decoded into the 4 numbers
(taxi_row, taxi_col, passenger_location, destination) and fed to the network.

    python src/dqn_taxi.py train            # train from scratch -> runs/
    python src/dqn_taxi.py test             # evaluate the pretrained model
    python src/dqn_taxi.py test --model runs/taxi_dqn.pth
"""

import argparse

import gymnasium as gym
import numpy as np

from common import MODELS_DIR, RUNS_DIR, plot_rewards, set_seed, summarize
from dqn import DQNAgent

ENV_ID = "Taxi-v3"
STATE_DIM = 4
N_ACTIONS = 6

# Hyper-parameters used for the pretrained model
HPARAMS = dict(
    lr=1e-3,
    gamma=0.99,
    batch_size=64,
    memory_size=100_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.99997,  # applied after every environment step
)


def encode(env, observation):
    """Integer Taxi state -> float vector [row, col, passenger, destination]."""
    return np.array(list(env.unwrapped.decode(observation)), dtype=np.float32)


def train(args):
    set_seed(args.seed)
    env = gym.make(ENV_ID)
    env.action_space.seed(args.seed)
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=args.seed, **HPARAMS)

    rewards = []
    for episode in range(args.episodes):
        # seed only the first reset; later resets continue the same RNG stream
        observation, info = env.reset(seed=args.seed if episode == 0 else None)
        state = encode(env, observation)
        episode_reward, done = 0.0, False
        while not done:
            action = agent.act(state, action_mask=info["action_mask"])
            agent.decay_epsilon()
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = encode(env, observation)
            agent.remember(state, action, reward, next_state, terminated)
            agent.learn()
            state = next_state
            episode_reward += reward
            done = terminated or truncated
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
    plot_rewards(rewards, 50, "Taxi - DQN training", RUNS_DIR / "taxi-dqn-training.png")


def test(args):
    set_seed(args.seed)
    env = gym.make(ENV_ID)
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=args.seed, **HPARAMS)
    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    rewards = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed if episode == 0 else None)
        episode_reward, done = 0.0, False
        while not done:
            action = agent.act(encode(env, observation), greedy=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    env.close()

    summarize(rewards)
    plot_rewards(rewards, 20, "Taxi - DQN testing", RUNS_DIR / "taxi-dqn-testing.png")


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
        args.model = args.model or RUNS_DIR / "taxi_dqn.pth"
        train(args)
    else:
        args.episodes = args.episodes or 1_000
        args.model = args.model or MODELS_DIR / "taxi_dqn.pth"
        test(args)


if __name__ == "__main__":
    main()
