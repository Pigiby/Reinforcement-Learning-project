"""Deep Q-Network agent shared by the Taxi and MountainCar experiments.

The agent follows the setup described in the report:
  * a fully connected network  input -> 128 -> 128 -> 128 -> n_actions  (ReLU),
  * an experience replay buffer sampled uniformly at random,
  * an epsilon-greedy behaviour policy with multiplicative epsilon decay,
  * MSE loss on the one-step Bellman target, optimised with Adam.

There is no separate target network: the same network produces both the
prediction and the bootstrap target, exactly as in the original experiments.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    """Maps a state vector to one Q-value per action.

    The layer names (fc1..fc4) must stay as they are so that the checkpoints
    in ``models/`` keep loading.
    """

    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', terminated) transitions."""

    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.bool_)
        self.counter = 0  # total number of transitions ever stored

    def __len__(self):
        return min(self.counter, self.capacity)

    def store(self, state, action, reward, next_state, terminated):
        i = self.counter % self.capacity  # overwrite the oldest entry when full
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminals[i] = terminated
        self.counter += 1

    def sample(self, batch_size, rng):
        idx = rng.choice(len(self), batch_size, replace=False)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.terminals[idx]),
        )


class DQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        memory_size=100_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.99997,
        seed=None,
        device="cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.q_net = DeepQNetwork(state_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(memory_size, state_dim)

    def act(self, state, greedy=False, action_mask=None):
        """Epsilon-greedy action selection.

        ``action_mask`` (a 0/1 array, as returned by Taxi in ``info``) restricts
        random exploration to the actions that actually do something.
        """
        if not greedy and self.rng.random() < self.epsilon:
            if action_mask is not None:
                return int(self.rng.choice(np.flatnonzero(action_mask)))
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q_values = self.q_net(torch.as_tensor(state, device=self.device))
        return int(torch.argmax(q_values).item())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.store(state, action, reward, next_state, terminated)

    def learn(self):
        """One gradient step on a random minibatch from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, terminals = (
            t.to(self.device) for t in self.memory.sample(self.batch_size, self.rng)
        )

        # Q(s, a) for the actions that were actually taken
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + gamma * max_a' Q(s', a'), with no bootstrap
        # from terminal states. Time-limit truncation is NOT terminal.
        with torch.no_grad():
            q_next = self.q_net(next_states).max(dim=1).values
            q_next[terminals] = 0.0
            q_target = rewards + self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(state_dict)
