from typing import Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np


DEFAULT_ENV_NAME = "mm_default"

_DEFAULT_P = np.array(
    [
        [
            [0.2, 0.3, 0.3, 0.2],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
    ],
    dtype=np.float32,
)

_M10S10_P = np.array(
    [
        [0.05, 0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.00, 0.00],
        [0.00, 0.05, 0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.00],
        [0.00, 0.00, 0.05, 0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05],
        [0.00, 0.00, 0.00, 0.05, 0.35, 0.20, 0.15, 0.10, 0.10, 0.05],
        [0.00, 0.00, 0.00, 0.00, 0.05, 0.35, 0.25, 0.15, 0.10, 0.10],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.35, 0.35, 0.15, 0.10],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.40, 0.40, 0.15],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.50, 0.40],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.70],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ],
    dtype=np.float32,
)

ENV_SPECS: Dict[str, Dict[str, object]] = {
    DEFAULT_ENV_NAME: {
        "state_values": np.array([0, 1, 2, 3], dtype=np.int32),
        "failed_state": 3,
        "transition_no_maintenance": _DEFAULT_P[0],
        "cpm_mode": "linear",
        "cpm_base": 60.0,
        "cpm_slope": 10.0,
        "ccm": 250.0,
        "cs": 30.0,
        "threshold": 2,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 0.0,
    },
    "m10s10": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 20.0,
        "ccm": 100.0,
        "cs": 30.0,
        "threshold": 5,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": -100.0,
        "operating_reward_slope": 10.0,
    },
    "m10s10_poscost": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 100.0,
        "ccm": 200.0,
        "cs": 50.0,
        "threshold": 5,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 10.0,
    },
    "m10s10_poscost_v2": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 500.0,
        "ccm": 1000.0,
        "cs": 500.0,
        "threshold": 5,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 10.0,
    },
    "m10s10_industrial": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 100.0,
        "ccm": 250.0,
        "threshold": 4,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 0.0,
        "window_period": 5,
        "window_open_phase": 0,
        "include_window_phase": True,
        "capacity_normal": 2,
        "capacity_window": 10,
        "cs_normal": 160.0,
        "cs_window": 30.0,
        "train_groups": ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
        "operating_cost_mode": "industrial_two_train",
    },
}

S = np.array(ENV_SPECS[DEFAULT_ENV_NAME]["state_values"], copy=True)


def get_env_spec(env_name: str = DEFAULT_ENV_NAME) -> Dict[str, object]:
    if env_name not in ENV_SPECS:
        raise ValueError(f"Unknown env_name='{env_name}'. Available: {sorted(ENV_SPECS)}")
    return ENV_SPECS[env_name]


def get_state_values(env_name: str = DEFAULT_ENV_NAME) -> np.ndarray:
    return np.array(get_env_spec(env_name)["state_values"], copy=True)


def get_num_components(env_name: str, num_machines: int) -> int:
    spec = get_env_spec(env_name)
    groups = spec.get("train_groups")
    if groups:
        max_index = max(max(group) for group in groups)
        return max(max_index + 1, num_machines)
    return num_machines


def _uses_window_phase(spec: Dict[str, object]) -> bool:
    return bool(spec.get("include_window_phase", False))


def observation_dim(num_machines: int, env_name: str = DEFAULT_ENV_NAME) -> int:
    spec = get_env_spec(env_name)
    return int(num_machines + 1) if _uses_window_phase(spec) else int(num_machines)


def split_observation(
    observation: Sequence[int],
    num_machines: Optional[int] = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> Tuple[np.ndarray, Optional[int]]:
    spec = get_env_spec(env_name)
    obs_arr = np.asarray(observation, dtype=np.int32).reshape(-1)
    if not _uses_window_phase(spec):
        machine_dim = int(obs_arr.shape[0] if num_machines is None else num_machines)
        return obs_arr[:machine_dim].astype(np.int32), None

    machine_dim = int(obs_arr.shape[0] - 1 if num_machines is None else num_machines)
    return obs_arr[:machine_dim].astype(np.int32), int(obs_arr[machine_dim])


def make_observation(
    component_states: Sequence[int],
    window_phase: Optional[int],
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    component_arr = np.asarray(component_states, dtype=np.int32).reshape(-1)
    if not _uses_window_phase(spec):
        return component_arr.astype(np.int32)
    phase = 0 if window_phase is None else int(window_phase)
    return np.concatenate([component_arr, np.asarray([phase], dtype=np.int32)]).astype(np.int32)


def sample_initial_observation(
    num_machines: int,
    rng: Optional[np.random.RandomState] = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    rng = np.random if rng is None else rng
    component_states = rng.choice(get_state_values(env_name), size=num_machines).astype(np.int32)
    if not _uses_window_phase(spec):
        return component_states
    period = int(spec["window_period"])
    phase = int(rng.randint(period))
    return make_observation(component_states, phase, env_name=env_name)


def _window_phase_from_observation(
    observation: Sequence[int],
    env_name: str = DEFAULT_ENV_NAME,
) -> Optional[int]:
    _, phase = split_observation(observation, env_name=env_name)
    return phase


def _window_open(phase: Optional[int], spec: Dict[str, object]) -> bool:
    if not _uses_window_phase(spec):
        return False
    return int(phase if phase is not None else 0) == int(spec["window_open_phase"])


def _next_window_phase(phase: Optional[int], spec: Dict[str, object]) -> Optional[int]:
    if not _uses_window_phase(spec):
        return None
    period = int(spec["window_period"])
    curr = int(phase if phase is not None else 0)
    return int((curr + 1) % period)


def _maintenance_capacity_for_phase(phase: Optional[int], spec: Dict[str, object]) -> Optional[int]:
    if "capacity_normal" not in spec:
        return None
    if _window_open(phase, spec):
        return int(spec["capacity_window"])
    return int(spec["capacity_normal"])


def _setup_cost_for_phase(phase: Optional[int], spec: Dict[str, object]) -> float:
    if "cs_normal" not in spec:
        return float(spec["cs"])
    if _window_open(phase, spec):
        return float(spec["cs_window"])
    return float(spec["cs_normal"])


def _operating_cost(component_states: np.ndarray, spec: Dict[str, object]) -> float:
    mode = str(spec.get("operating_cost_mode", "per_component_linear"))
    if mode == "per_component_linear":
        intercept = float(spec["operating_reward_intercept"])
        slope = float(spec["operating_reward_slope"])
        return float(np.sum(intercept + slope * component_states.astype(np.float32)))

    if mode == "industrial_two_train":
        groups = spec["train_groups"]
        failed_state = int(spec["failed_state"])
        h_a = int(np.max(component_states[list(groups[0])]))
        h_b = int(np.max(component_states[list(groups[1])]))
        u_a = int(h_a < failed_state)
        u_b = int(h_b < failed_state)
        if u_a == 1 and u_b == 1:
            return float(10.0 * min(h_a, h_b))
        if u_a + u_b == 1:
            return float(120.0 + 15.0 * max(h_a, h_b))
        return 400.0

    raise ValueError(f"Unsupported operating_cost_mode='{mode}'")


def _preventive_cost(states_arr: np.ndarray, spec: Dict[str, object]) -> np.ndarray:
    mode = str(spec["cpm_mode"])
    if mode == "linear":
        return float(spec["cpm_base"]) + float(spec["cpm_slope"]) * states_arr.astype(np.float32)
    if mode == "constant":
        return np.full(states_arr.shape, float(spec["cpm_constant"]), dtype=np.float32)
    raise ValueError(f"Unsupported cpm_mode='{mode}'")


def cost_function(states: Sequence[int], actions: Sequence[int], env_name: str = DEFAULT_ENV_NAME) -> float:
    spec = get_env_spec(env_name)
    failed_state = int(spec["failed_state"])
    ccm = float(spec["ccm"])
    component_states, phase = split_observation(states, env_name=env_name)
    actions_arr = np.asarray(actions, dtype=np.int32)
    cpm = _preventive_cost(component_states, spec)
    y = (actions_arr == 1) & (component_states != failed_state)
    z = (actions_arr == 1) & (component_states == failed_state)
    operating_cost = _operating_cost(component_states, spec)
    cost = operating_cost + np.sum(cpm * y + ccm * z)
    if np.any(actions_arr):
        cost += _setup_cost_for_phase(phase, spec)
    return float(cost)


def take_step(
    states: Sequence[int],
    actions: Sequence[int],
    rng: Optional[np.random.RandomState] = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> Tuple[np.ndarray, float]:
    spec = get_env_spec(env_name)
    rng = np.random if rng is None else rng
    component_states, phase = split_observation(states, env_name=env_name)
    actions_arr = project_action(states, actions, env_name=env_name)
    state_values = np.asarray(spec["state_values"], dtype=np.int32)
    transition_no_maintenance = np.asarray(spec["transition_no_maintenance"], dtype=np.float32)

    next_states = []
    for state, action in zip(component_states, actions_arr):
        if action == 1:
            next_states.append(0)
        else:
            next_states.append(rng.choice(state_values, p=transition_no_maintenance[state]))
    cost = cost_function(states, actions_arr, env_name=env_name)
    next_phase = _next_window_phase(phase, spec)
    return make_observation(next_states, next_phase, env_name=env_name), cost


def random_viable_action(
    states: Sequence[int],
    rng: Optional[np.random.RandomState] = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    rng = np.random if rng is None else rng
    failed_state = int(spec["failed_state"])
    enforce_zero = bool(spec["enforce_zero_no_maintenance"])
    actions: List[int] = []
    component_states, _ = split_observation(states, env_name=env_name)
    for state in component_states:
        if state == failed_state:
            actions.append(1)
        elif enforce_zero and state == 0:
            actions.append(0)
        else:
            actions.append(int(rng.choice([0, 1])))
    return np.asarray(actions, dtype=np.int32)


def threshold_policy(
    states: Sequence[int],
    threshold: Optional[int] = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    threshold_value = int(spec["threshold"] if threshold is None else threshold)
    component_states, _ = split_observation(states, env_name=env_name)
    actions = np.asarray(
        [1 if int(state) >= threshold_value else 0 for state in component_states],
        dtype=np.int32,
    )
    return project_action(states, actions, env_name=env_name)


def train_priority_threshold_policy(
    states: Sequence[int],
    threshold: Optional[int] = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    component_states, phase = split_observation(states, env_name=env_name)
    threshold_value = int(spec["threshold"] if threshold is None else threshold)
    failed_state = int(spec["failed_state"])
    actions = np.zeros(component_states.shape[0], dtype=np.int32)

    # Failed components are always replaced.
    actions[component_states == failed_state] = 1

    groups = spec.get("train_groups")
    if not groups:
        candidate_idx = np.where(
            (component_states >= threshold_value) & (component_states != failed_state)
        )[0]
        if candidate_idx.size:
            order = candidate_idx[np.argsort(component_states[candidate_idx])[::-1]]
            actions[order] = 1
        return project_action(states, actions, env_name=env_name)

    priority_groups = sorted(
        groups,
        key=lambda group: int(np.max(component_states[list(group)])),
        reverse=True,
    )

    capacity = _maintenance_capacity_for_phase(phase, spec)
    remaining_preventive = None if capacity is None else int(capacity)

    for group in priority_groups:
        candidate_idx = np.asarray(
            [
                idx for idx in group
                if component_states[idx] >= threshold_value and component_states[idx] != failed_state
            ],
            dtype=np.int32,
        )
        if candidate_idx.size == 0:
            continue
        order = candidate_idx[np.argsort(component_states[candidate_idx])[::-1]]
        if remaining_preventive is None:
            actions[order] = 1
            continue
        if remaining_preventive <= 0:
            break
        keep = order[:remaining_preventive]
        actions[keep] = 1
        remaining_preventive -= int(keep.size)

    return project_action(states, actions, env_name=env_name)


def window_batch_policy(
    states: Sequence[int],
    normal_threshold: int = 7,
    window_threshold: int = 5,
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    component_states, phase = split_observation(states, env_name=env_name)
    failed_state = int(spec["failed_state"])
    actions = np.zeros(component_states.shape[0], dtype=np.int32)
    actions[component_states == failed_state] = 1

    capacity = _maintenance_capacity_for_phase(phase, spec)
    is_window = _window_open(phase, spec)
    threshold_value = int(window_threshold if is_window else normal_threshold)

    candidate_idx = np.where(
        (component_states >= threshold_value) & (component_states != failed_state)
    )[0]
    if candidate_idx.size:
        order = candidate_idx[np.argsort(component_states[candidate_idx])[::-1]]
        if capacity is None:
            actions[order] = 1
        elif is_window:
            actions[order[:int(capacity)]] = 1
        else:
            actions[order[:int(capacity)]] = 1

    return project_action(states, actions, env_name=env_name)


def named_policy(
    states: Sequence[int],
    policy_name: str,
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    if policy_name.startswith("component_threshold_t"):
        threshold = int(policy_name.split("component_threshold_t", 1)[1])
        return threshold_policy(states, threshold=threshold, env_name=env_name)
    if policy_name.startswith("train_priority_t"):
        threshold = int(policy_name.split("train_priority_t", 1)[1])
        return train_priority_threshold_policy(states, threshold=threshold, env_name=env_name)
    if policy_name.startswith("window_batch_n"):
        suffix = policy_name.split("window_batch_n", 1)[1]
        normal_raw, window_raw = suffix.split("_w", 1)
        return window_batch_policy(
            states,
            normal_threshold=int(normal_raw),
            window_threshold=int(window_raw),
            env_name=env_name,
        )
    raise ValueError(f"Unsupported policy_name='{policy_name}'")


def project_action(
    states: Sequence[int],
    actions: Sequence[int],
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    failed_state = int(spec["failed_state"])
    enforce_zero = bool(spec["enforce_zero_no_maintenance"])
    component_states, phase = split_observation(states, env_name=env_name)
    actions_arr = np.asarray(actions, dtype=np.int32).reshape(-1)
    if actions_arr.shape[0] != component_states.shape[0]:
        actions_arr = actions_arr[:component_states.shape[0]]
    corrected = []
    for state, action in zip(component_states, actions_arr):
        if state == failed_state:
            corrected.append(1)
        elif enforce_zero and state == 0:
            corrected.append(0)
        else:
            corrected.append(int(np.clip(action, 0, 1)))
    corrected_arr = np.asarray(corrected, dtype=np.int32)

    capacity = _maintenance_capacity_for_phase(phase, spec)
    if capacity is not None:
        preventive_selected = np.where(
            (corrected_arr == 1) & (component_states != failed_state)
        )[0]
        if preventive_selected.size > capacity:
            priority = np.argsort(component_states[preventive_selected])[::-1]
            keep = preventive_selected[priority[:capacity]]
            drop = set(preventive_selected.tolist()) - set(keep.tolist())
            for idx in drop:
                corrected_arr[idx] = 0
    return corrected_arr


class MachineMaintenanceEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        num_machines: int = 25,
        horizon: int = 100,
        reward_is_negative_cost: bool = True,
        seed: Optional[int] = None,
        env_name: str = DEFAULT_ENV_NAME,
    ):
        super().__init__()
        self.num_machines = num_machines
        self.horizon = horizon
        self.reward_is_negative_cost = reward_is_negative_cost
        self.env_name = env_name
        self.spec = get_env_spec(env_name)
        self.state_values = np.asarray(self.spec["state_values"], dtype=np.int32)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=float(max(self.state_values.max(), int(self.spec.get("window_period", 1)) - 1)),
            shape=(observation_dim(num_machines, env_name=env_name),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiBinary(num_machines)

        self._rng = np.random.RandomState(seed)
        self._state = None
        self._t = 0
        self._episode_return = 0.0
        self._episode_cost = 0.0

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self._t = 0
        self._episode_return = 0.0
        self._episode_cost = 0.0
        self._state = sample_initial_observation(
            self.num_machines, rng=self._rng, env_name=self.env_name
        )
        return self._state.astype(np.float32)

    def step(self, action):
        if self._state is None:
            raise ValueError("Environment must be reset before calling step().")

        action_arr = np.asarray(action, dtype=np.int32).reshape(self.num_machines)
        action_arr = project_action(self._state, action_arr, env_name=self.env_name)

        next_state, cost = take_step(self._state, action_arr, rng=self._rng, env_name=self.env_name)
        reward = -cost if self.reward_is_negative_cost else cost

        self._state = next_state
        self._t += 1
        self._episode_return += reward
        self._episode_cost += cost

        done = self._t >= self.horizon
        info: Dict[str, Dict[str, float]] = {}
        if done:
            info["episode"] = {
                "return": self._episode_return,
                "length": self._t,
                "cost": self._episode_cost,
            }

        return self._state.astype(np.float32), float(reward), done, info
