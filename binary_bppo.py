from copy import deepcopy

import torch

from buffer import OnlineReplayBuffer
from critic import QLearner, ValueLearner
from net import BernoulliPolicyMLP
from utils import CONST_EPS, log_prob_func, orthogonal_initWeights


class BinaryPolicyBase:
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int,
        depth: int,
        action_dim: int,
        policy_lr: float,
        batch_size: int,
    ) -> None:
        self._device = device
        self._policy = BernoulliPolicyMLP(
            state_dim, hidden_dim, depth, action_dim).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(), lr=policy_lr)
        self._policy_lr = policy_lr
        self._batch_size = batch_size

    def select_action(self, s: torch.Tensor, is_sample: bool) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:
            action = (dist.probs >= 0.5).float()
        return action

    def save(self, path: str) -> None:
        torch.save(self._policy.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self._device)
        self._policy.load_state_dict(state_dict)


class BehaviorCloning(BinaryPolicyBase):
    def loss(self, replay_buffer: OnlineReplayBuffer) -> torch.Tensor:
        s, a, _, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        dist = self._policy(s)
        log_prob = log_prob_func(dist, a)
        return (-log_prob).mean()

    def update(self, replay_buffer: OnlineReplayBuffer) -> float:
        policy_loss = self.loss(replay_buffer)
        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
        return policy_loss.item()


class ProximalPolicyOptimization(BinaryPolicyBase):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int,
        depth: int,
        action_dim: int,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
    ) -> None:
        super().__init__(
            device=device,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            action_dim=action_dim,
            policy_lr=policy_lr,
            batch_size=batch_size,
        )
        self._old_policy = deepcopy(self._policy)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=2, gamma=0.98)
        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._decay = decay
        self._omega = omega

    def weighted_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        weight = torch.zeros_like(advantage)
        positive_index = torch.where(advantage > 0)[0]
        weight[positive_index] = self._omega
        weight[torch.where(weight == 0)[0]] = 1 - self._omega
        return weight.to(self._device) * advantage

    def set_old_policy(self) -> None:
        self._old_policy.load_state_dict(self._policy.state_dict())

    def load(self, path: str) -> None:
        super().load(path)
        self._old_policy.load_state_dict(self._policy.state_dict())

    def loss(
        self,
        replay_buffer: OnlineReplayBuffer,
        q_fn: QLearner,
        value_fn: ValueLearner,
        is_clip_decay: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    def update(
        self,
        replay_buffer: OnlineReplayBuffer,
        q_fn: QLearner,
        value_fn: ValueLearner,
        is_clip_decay: bool,
        is_lr_decay: bool,
    ) -> float:
        policy_loss = self.loss(replay_buffer, q_fn, value_fn, is_clip_decay)
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
        self._optimizer.step()
        if is_lr_decay:
            self._scheduler.step()
        return policy_loss.item()


class BehaviorProximalPolicyOptimization(ProximalPolicyOptimization):
    def loss(
        self,
        replay_buffer: OnlineReplayBuffer,
        q_fn: QLearner,
        value_fn: ValueLearner,
        is_clip_decay: bool,
    ) -> torch.Tensor:
        s, _, _, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        old_dist = self._old_policy(s)
        a = old_dist.sample()
        advantage = q_fn(s, a) - value_fn(s)
        advantage = (advantage - advantage.mean()) / (
            advantage.std() + CONST_EPS)

        new_dist = self._policy(s)
        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
        advantage = self.weighted_advantage(advantage)
        loss1 = ratio * advantage
        if is_clip_decay:
            self._clip_ratio = self._clip_ratio * self._decay
        loss2 = torch.clamp(
            ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage
        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        return -(torch.min(loss1, loss2) + entropy_loss).mean()
