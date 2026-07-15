"""TD3+BC 에이전트 (Fujimoto & Gu, 2021).

참고 구현:
  - sfujim/TD3_BC   (원조)      https://github.com/sfujim/TD3_BC
  - corl-team/CORL  (단일파일)  https://github.com/corl-team/CORL

TD3 (Twin Delayed DDPG) + Behavior Cloning 규제:
  actor_loss = -lambda * Q(s, pi(s)) + MSE(pi(s), a)
  lambda      = alpha / mean(|Q|)   ← Q 크기에 맞춰 BC와 RL의 균형을 자동 조절
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1))
        return self.l3(q1)


class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action, device,
                 discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2, alpha=2.5, lr=3e-4):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.actor(state).cpu().numpy().flatten()

    def train(self, buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            tq1, tq2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(tq1, tq2)

        cq1, cq2 = self.critic(state, action)
        critic_loss = F.mse_loss(cq1, target_Q) + F.mse_loss(cq2, target_Q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        info = {"critic_loss": float(critic_loss.item())}
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            bc = F.mse_loss(pi, action)
            actor_loss = -lmbda * Q.mean() + bc
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            info.update({"actor_loss": float(actor_loss.item()), "bc_loss": float(bc.item()),
                         "Q": float(Q.mean().item())})
        return info

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
