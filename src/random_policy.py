"""Uniform random policy, the baseline the trained agents are compared against.

For Taxi, actions are sampled only among the valid ones (``info["action_mask"]``),
so the random taxi never attempts an illegal pickup/drop-off.

    python src/random_policy.py MountainCar-v0
    python src/random_policy.py Taxi-v3
"""

import argparse

import gymnasium as gym

from common import RUNS_DIR, plot_rewards, set_seed, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env", choices=["MountainCar-v0", "Taxi-v3"])
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    rewards = []
    for episode in range(args.episodes):
        _, info = env.reset(seed=args.seed if episode == 0 else None)
        episode_reward, done = 0.0, False
        while not done:
            action = env.action_space.sample(info.get("action_mask"))
            _, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    env.close()

    summarize(rewards)
    name = args.env.split("-")[0].lower()
    plot_rewards(rewards, 20, f"{args.env} - random policy", RUNS_DIR / f"{name}-random-testing.png")


if __name__ == "__main__":
    main()
