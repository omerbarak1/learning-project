# agents/deep_prompt_agent.py

import random, collections
import torch
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition: (state, action, tmpl_idx, reward, next_state)"""
        self.buffer.append(tuple(args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128,      128), nn.ReLU(),
            nn.Linear(128,      output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DeepPromptAgent:
    def __init__(self,
                 state_dim,
                 n_actions,
                 templates_per_action,
                 gamma=0.99,
                 lr=1e-3,
                 eps_start=1.0,
                 eps_min=0.05,
                 eps_decay=0.995,
                 buffer_capacity=5000,
                 batch_size=64,
                 target_update_freq=500):
        # --- action Q‑network ---
        self.q_action_policy = MLP(state_dim, n_actions)
        self.q_action_target = MLP(state_dim, n_actions)
        self.opt_action       = optim.Adam(self.q_action_policy.parameters(), lr=lr)

        # --- template Q‑networks (one head per action) ---
        self.q_tmpl_policy = {
            a: MLP(state_dim, k)
            for a, k in enumerate(templates_per_action)
        }
        self.q_tmpl_target = {
            a: MLP(state_dim, k)
            for a, k in enumerate(templates_per_action)
        }
        self.opt_tmpl = {
            a: optim.Adam(self.q_tmpl_policy[a].parameters(), lr=lr)
            for a in self.q_tmpl_policy
        }

        # experience replay
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        # ε‑greedy
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # target update
        self.target_update_freq = target_update_freq
        self.step_count = 0

    def select(self, state_vec):
        """Return (action_idx, template_idx) using ε‑greedy on both networks."""
        s = torch.FloatTensor(state_vec).unsqueeze(0)  # shape [1, state_dim]

        # action
        if random.random() < self.eps:
            action = random.randrange(self.q_action_policy.net[-1].out_features)
        else:
            qv = self.q_action_policy(s)
            action = qv.argmax().item()

        # template
        q_tmpl = self.q_tmpl_policy[action](s)
        if random.random() < self.eps:
            tmpl_idx = random.randrange(q_tmpl.size(1))
        else:
            tmpl_idx = q_tmpl.argmax().item()

        return action, tmpl_idx

    def push_transition(self, state, action, tmpl_idx, reward, next_state):
        self.buffer.push(state, action, tmpl_idx, reward, next_state)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, tmpls, rewards, next_states = zip(*batch)

        S  = torch.FloatTensor(states)
        S2 = torch.FloatTensor(next_states)
        R  = torch.FloatTensor(rewards).unsqueeze(1)

        # --- update action net ---
        Q1     = self.q_action_policy(S).gather(1, torch.LongTensor(actions).unsqueeze(1))
        Q1_next= self.q_action_target(S2).max(1)[0].detach().unsqueeze(1)
        Y1     = R + self.gamma * Q1_next
        loss1  = nn.MSELoss()(Q1, Y1)

        self.opt_action.zero_grad()
        loss1.backward()
        self.opt_action.step()

        # --- update template nets ---
        for a in set(actions):
            idxs = [i for i, aa in enumerate(actions) if aa == a]
            Sa   = S[idxs]
            Ka   = torch.LongTensor([tmpls[i] for i in idxs]).unsqueeze(1)
            Ra   = R[idxs]
            S2a  = S2[idxs]

            Q2      = self.q_tmpl_policy[a](Sa).gather(1, Ka)
            Q2_next = self.q_tmpl_target[a](S2a).max(1)[0].detach().unsqueeze(1)
            Y2      = Ra + self.gamma * Q2_next
            loss2   = nn.MSELoss()(Q2, Y2)

            opt = self.opt_tmpl[a]
            opt.zero_grad()
            loss2.backward()
            opt.step()

        # ε‑decay
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        self.step_count += 1

        # sync targets
        if self.step_count % self.target_update_freq == 0:
            self.q_action_target.load_state_dict(self.q_action_policy.state_dict())
            for a in self.q_tmpl_policy:
                self.q_tmpl_target[a].load_state_dict(self.q_tmpl_policy[a].state_dict())
