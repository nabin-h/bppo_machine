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

_GAMMA_RUL_P = np.array(
    [
        [0.911792568083927, 0.08626750556593676, 0.001856691563573154, 7.916261159657267e-05, 3.860117807508168e-06, 2.005989783171458e-07, 1.0823697271789001e-08, 5.987301765486563e-10, 3.371192214274288e-11, 2.0413670753782753e-12],
        [0.0, 0.911792568083927, 0.08626750556593676, 0.001856691563573154, 7.916261159657267e-05, 3.860117807508168e-06, 2.005989783171458e-07, 1.0823697271789001e-08, 5.987301765486563e-10, 3.5753289218121154e-11],
        [0.0, 0.0, 0.911792568083927, 0.08626750556593676, 0.001856691563573154, 7.916261159657267e-05, 3.860117807508168e-06, 2.005989783171458e-07, 1.0823697271789001e-08, 6.344834657667775e-10],
        [0.0, 0.0, 0.0, 0.911792568083927, 0.08626750556593676, 0.001856691563573154, 7.916261159657267e-05, 3.860117807508168e-06, 2.005989783171458e-07, 1.1458180737555779e-08],
        [0.0, 0.0, 0.0, 0.0, 0.911792568083927, 0.08626750556593676, 0.001856691563573154, 7.916261159657267e-05, 3.860117807508168e-06, 2.1205715905470157e-07],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.911792568083927, 0.08626750556593676, 0.001856691563573154, 7.916261159657267e-05, 4.072174966562869e-06],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.911792568083927, 0.08626750556593676, 0.001856691563573154, 8.323478656313554e-05],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.911792568083927, 0.08626750556593676, 0.0019399263501362896],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.911792568083927, 0.08820743191607305],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_GAMMA_RUL_EXPECTED_LIFE = np.array(
    [
        99.99999999990796,
        88.91781209589504,
        77.83562419184308,
        66.75343628703588,
        55.671248367147676,
        44.589060132535316,
        33.50686488821598,
        22.424496217270452,
        11.33691320875856,
        0.0,
    ],
    dtype=np.float64,
)

_GAMMA_V2_FAST_P = np.array(
    [
        [0.7922032506223413, 0.1668662699425315, 0.02910779906477523, 0.008071154012708526, 0.0025027837865402303, 0.0008206374597989896, 0.00027834956392258636, 9.659372132808652e-05, 3.407697312340918e-05, 1.9084852930117968e-05],
        [0.0, 0.7922032506223413, 0.1668662699425315, 0.02910779906477523, 0.008071154012708526, 0.0025027837865402303, 0.0008206374597989896, 0.00027834956392258636, 9.659372132808652e-05, 5.316182605352715e-05],
        [0.0, 0.0, 0.7922032506223413, 0.1668662699425315, 0.02910779906477523, 0.008071154012708526, 0.0025027837865402303, 0.0008206374597989896, 0.00027834956392258636, 0.00014975554738161367],
        [0.0, 0.0, 0.0, 0.7922032506223413, 0.1668662699425315, 0.02910779906477523, 0.008071154012708526, 0.0025027837865402303, 0.0008206374597989896, 0.00042810511130420004],
        [0.0, 0.0, 0.0, 0.0, 0.7922032506223413, 0.1668662699425315, 0.02910779906477523, 0.008071154012708526, 0.0025027837865402303, 0.0012487425711031896],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.7922032506223413, 0.1668662699425315, 0.02910779906477523, 0.008071154012708526, 0.00375152635764342],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7922032506223413, 0.1668662699425315, 0.02910779906477523, 0.011822680370351946],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7922032506223413, 0.1668662699425315, 0.040930479435127176],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7922032506223413, 0.2077967493776587],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_GAMMA_V3_P = np.array(
    [
        [0.792, 0.167, 0.029, 0.008, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.792, 0.167, 0.029, 0.008, 0.003, 0.001, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.792, 0.167, 0.029, 0.008, 0.003, 0.001, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.792, 0.167, 0.029, 0.008, 0.003, 0.001, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.792, 0.167, 0.029, 0.008, 0.003, 0.001],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.792, 0.167, 0.029, 0.008, 0.004],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.792, 0.167, 0.029, 0.012],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.792, 0.167, 0.041],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.792, 0.208],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    ],
    dtype=np.float64,
)

_GAMMA_V3_HETERO_SLOW_P = np.array(
    [
        [0.826, 0.150, 0.019, 0.004, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.826, 0.150, 0.019, 0.004, 0.001, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.826, 0.150, 0.019, 0.004, 0.001, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.826, 0.150, 0.019, 0.004, 0.001, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.826, 0.150, 0.019, 0.004, 0.001, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.826, 0.150, 0.019, 0.004, 0.001],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.826, 0.150, 0.019, 0.005],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.826, 0.150, 0.024],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.826, 0.174],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    ],
    dtype=np.float64,
)

_GAMMA_V3_HETERO_FAST_P = np.array(
    [
        [0.760, 0.179, 0.039, 0.013, 0.005, 0.002, 0.001, 0.000, 0.000, 0.001],
        [0.000, 0.760, 0.179, 0.039, 0.013, 0.005, 0.002, 0.001, 0.000, 0.001],
        [0.000, 0.000, 0.760, 0.179, 0.039, 0.013, 0.005, 0.002, 0.001, 0.001],
        [0.000, 0.000, 0.000, 0.760, 0.179, 0.039, 0.013, 0.005, 0.002, 0.002],
        [0.000, 0.000, 0.000, 0.000, 0.760, 0.179, 0.039, 0.013, 0.005, 0.004],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.760, 0.179, 0.039, 0.013, 0.009],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.760, 0.179, 0.039, 0.022],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.760, 0.179, 0.061],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.760, 0.240],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    ],
    dtype=np.float64,
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
    "m10s10_env3": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 60.0,
        "ccm": 120.0,
        "cs": 80.0,
        "threshold": 6,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 3.0,
    },
    "env3": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 60.0,
        "ccm": 120.0,
        "cs": 80.0,
        "threshold": 6,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 3.0,
    },
    "m20s10_env3": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 60.0,
        "ccm": 120.0,
        "cs": 80.0,
        "threshold": 6,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 3.0,
    },
    "gamma_rul_v1": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _GAMMA_RUL_P,
        "cpm_mode": "remaining_life",
        "cpm_base": 60.0,
        "cpm_rul_rate": 0.6,
        "expected_remaining_life": _GAMMA_RUL_EXPECTED_LIFE,
        "ccm": 120.0,
        "cs": 80.0,
        "threshold": 8,
        "opportunity_trigger": 8,
        "opportunity_threshold": 6,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 1.0,
        "initialization_mode": "threshold_burnin",
        "initialization_thresholds": (5, 6, 7),
        "initialization_burnin_steps": 500,
        "vectorized_component_transitions": True,
        "degradation_model": {
            "type": "stationary_gamma_process_discretization",
            "shape_rate": 0.25,
            "scale": 0.3609395576619267,
            "failure_threshold": 9.0,
            "time_step": 1.0,
            "target_mttf": 100.0,
        },
    },
    "gamma_v2": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _GAMMA_RUL_P,
        "cpm_mode": "constant",
        "cpm_constant": 90.0,
        "ccm": 200.0,
        "cs": 80.0,
        "threshold": 6,
        "opportunity_trigger": 7,
        "opportunity_threshold": 4,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 1.5,
        "operating_cost_action0_only": True,
        "initialization_mode": "threshold_burnin",
        "initialization_thresholds": (5, 6, 7),
        "initialization_burnin_steps": 500,
        "vectorized_component_transitions": True,
        "degradation_model": {
            "type": "stationary_gamma_process_discretization",
            "shape_rate": 0.25,
            "scale": 0.3609395576619267,
            "failure_threshold": 9.0,
            "time_step": 1.0,
            "target_mttf": 100.0,
        },
    },
    "gamma_v2_fast": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _GAMMA_V2_FAST_P,
        "cpm_mode": "constant",
        "cpm_constant": 90.0,
        "ccm": 200.0,
        "cs": 80.0,
        "threshold": 6,
        "opportunity_trigger": 8,
        "opportunity_threshold": 5,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 1.5,
        "operating_cost_action0_only": True,
        "initialization_mode": "threshold_burnin",
        "initialization_thresholds": (5, 6, 7),
        "initialization_burnin_steps": 500,
        "vectorized_component_transitions": True,
        "degradation_model": {
            "type": "stationary_gamma_process_discretization",
            "shape_rate": 0.25,
            "scale": 1.0648447491773498,
            "failure_threshold": 9.0,
            "time_step": 1.0,
            "target_mttf": 35.0,
        },
    },
    "gamma_v3": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _GAMMA_V3_P,
        "cpm_mode": "constant",
        "cpm_constant": 100.0,
        "ccm": 200.0,
        "cs": 100.0,
        "threshold": 6,
        "opportunity_trigger": 8,
        "opportunity_threshold": 5,
        "enforce_zero_no_maintenance": True,
        "operating_reward_intercept": 0.0,
        "operating_reward_slope": 2.0,
        "operating_cost_action0_only": True,
        "initialization_mode": "threshold_burnin",
        "initialization_thresholds": (5, 6, 7),
        "initialization_burnin_steps": 500,
        "vectorized_component_transitions": True,
        "degradation_model": {
            "type": "rounded_stationary_gamma_process_discretization",
            "shape_rate": 0.25,
            "scale": 1.0648447491773498,
            "failure_threshold": 9.0,
            "time_step": 1.0,
            "target_mttf": 35.0,
            "discrete_mttf": 34.979373761843966,
            "transition_decimals": 3,
        },
    },
    "m10s10_industrial": {
        "state_values": np.arange(10, dtype=np.int32),
        "failed_state": 9,
        "transition_no_maintenance": _M10S10_P,
        "cpm_mode": "constant",
        "cpm_constant": 100.0,
        "ccm": 250.0,
        "threshold": 5,
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

ENV_SPECS["gamma_v3_cs500"] = {**ENV_SPECS["gamma_v3"], "cs": 500.0}
ENV_SPECS["gamma_v3_cs1000"] = {**ENV_SPECS["gamma_v3"], "cs": 1000.0}
ENV_SPECS["gamma_v3_cs100"] = {**ENV_SPECS["gamma_v3"], "cs": 100.0}
ENV_SPECS["gamma_v3_cs200"] = {**ENV_SPECS["gamma_v3"], "cs": 200.0}

_GAMMA_V3_HETERO_TRANSITIONS = {
    "slow": _GAMMA_V3_HETERO_SLOW_P,
    "medium": _GAMMA_V3_P,
    "fast": _GAMMA_V3_HETERO_FAST_P,
}
_GAMMA_V3_HETERO_DEGRADATION = {
    "type": "rounded_stationary_gamma_process_discretization_by_class",
    "shape_rate": 0.25,
    "failure_threshold": 9.0,
    "time_step": 1.0,
    "transition_decimals": 3,
    "classes": {
        "slow": {
            "target_mttf": 45.0,
            "discrete_mttf": 45.00672803223509,
            "scale": 0.8172328301090045,
        },
        "medium": {
            "target_mttf": 35.0,
            "discrete_mttf": 34.979373761843966,
            "scale": 1.0648447491773498,
        },
        "fast": {
            "target_mttf": 28.0,
            "discrete_mttf": 27.93229306110133,
            "scale": 1.3532621987341726,
        },
    },
}
ENV_SPECS["gamma_v3_hetero_cs5n_m20"] = {
    **ENV_SPECS["gamma_v3_cs100"],
    "transition_no_maintenance_by_class": _GAMMA_V3_HETERO_TRANSITIONS,
    "component_classes": ("slow",) * 7 + ("medium",) * 7 + ("fast",) * 6,
    "expected_num_machines": 20,
    "degradation_model": _GAMMA_V3_HETERO_DEGRADATION,
}
ENV_SPECS["gamma_v3_hetero_cs5n_m40"] = {
    **ENV_SPECS["gamma_v3_cs200"],
    "transition_no_maintenance_by_class": _GAMMA_V3_HETERO_TRANSITIONS,
    "component_classes": ("slow",) * 14 + ("medium",) * 14 + ("fast",) * 12,
    "expected_num_machines": 40,
    "degradation_model": _GAMMA_V3_HETERO_DEGRADATION,
}

ENV_SPECS["gamma_v3_hetero_cs25n_m20"] = {
    **ENV_SPECS["gamma_v3_cs500"],
    "transition_no_maintenance_by_class": _GAMMA_V3_HETERO_TRANSITIONS,
    "component_classes": ("slow",) * 7 + ("medium",) * 7 + ("fast",) * 6,
    "expected_num_machines": 20,
    "degradation_model": _GAMMA_V3_HETERO_DEGRADATION,
}

ENV_SPECS["gamma_v3_hetero_cs25n_m40"] = {
    **ENV_SPECS["gamma_v3_cs1000"],
    "transition_no_maintenance_by_class": _GAMMA_V3_HETERO_TRANSITIONS,
    "component_classes": ("slow",) * 14 + ("medium",) * 14 + ("fast",) * 12,
    "expected_num_machines": 40,
    "degradation_model": _GAMMA_V3_HETERO_DEGRADATION,
}

S = np.array(ENV_SPECS[DEFAULT_ENV_NAME]["state_values"], copy=True)
_BURNIN_DISTRIBUTION_CACHE: Dict[Tuple[object, ...], np.ndarray] = {}
_COMPONENT_TRANSITION_CACHE: Dict[Tuple[str, int], np.ndarray] = {}


def get_env_spec(env_name: str = DEFAULT_ENV_NAME) -> Dict[str, object]:
    if env_name not in ENV_SPECS:
        raise ValueError(f"Unknown env_name='{env_name}'. Available: {sorted(ENV_SPECS)}")
    return ENV_SPECS[env_name]


def get_state_values(env_name: str = DEFAULT_ENV_NAME) -> np.ndarray:
    return np.array(get_env_spec(env_name)["state_values"], copy=True)


def component_transition_matrices(
    num_machines: int,
    env_name: str = DEFAULT_ENV_NAME,
) -> Optional[np.ndarray]:
    spec = get_env_spec(env_name)
    transitions = spec.get("transition_no_maintenance_by_class")
    classes = spec.get("component_classes")
    if transitions is None and classes is None:
        return None
    if transitions is None or classes is None:
        raise ValueError("Heterogeneous environments require transitions and classes.")
    if len(classes) != num_machines:
        raise ValueError(
            f"Environment '{env_name}' requires {len(classes)} machines, "
            f"not {num_machines}."
        )
    cache_key = (env_name, num_machines)
    cached = _COMPONENT_TRANSITION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    stacked = np.stack(
        [np.asarray(transitions[label], dtype=np.float64) for label in classes],
        axis=0,
    )
    stacked.setflags(write=False)
    _COMPONENT_TRANSITION_CACHE[cache_key] = stacked
    return stacked


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
    initialization_mode = str(spec.get("initialization_mode", "uniform"))
    if initialization_mode == "uniform":
        component_states = rng.choice(
            get_state_values(env_name), size=num_machines
        ).astype(np.int32)
    elif initialization_mode == "threshold_burnin":
        thresholds = np.asarray(spec["initialization_thresholds"], dtype=np.int32)
        threshold = int(rng.choice(thresholds))
        burnin_steps = int(spec["initialization_burnin_steps"])
        component_transitions = component_transition_matrices(
            num_machines, env_name=env_name
        )
        if component_transitions is None:
            component_transitions = np.repeat(
                np.asarray(spec["transition_no_maintenance"], dtype=np.float64)[
                    np.newaxis, :, :
                ],
                num_machines,
                axis=0,
            )
            component_classes = ("homogeneous",) * num_machines
        else:
            component_classes = tuple(spec["component_classes"])

        component_states = np.empty(num_machines, dtype=np.int32)
        state_values = get_state_values(env_name)
        for class_label in dict.fromkeys(component_classes):
            class_indices = np.flatnonzero(
                np.asarray(component_classes, dtype=object) == class_label
            )
            transition = component_transitions[class_indices[0]]
            cache_key = (env_name, class_label, threshold, burnin_steps)
            distribution = _BURNIN_DISTRIBUTION_CACHE.get(cache_key)
            if distribution is None:
                policy_transition = transition.copy()
                reset_row = np.zeros(transition.shape[1], dtype=np.float64)
                reset_row[0] = 1.0
                policy_transition[threshold:] = reset_row
                distribution = np.zeros(transition.shape[0], dtype=np.float64)
                distribution[0] = 1.0
                for _ in range(burnin_steps):
                    distribution = distribution @ policy_transition
                distribution /= distribution.sum()
                _BURNIN_DISTRIBUTION_CACHE[cache_key] = distribution
            component_states[class_indices] = rng.choice(
                state_values,
                size=class_indices.size,
                p=distribution,
            ).astype(np.int32)
    else:
        raise ValueError(f"Unsupported initialization_mode='{initialization_mode}'")
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


def _operating_cost(
    component_states: np.ndarray,
    spec: Dict[str, object],
    actions: Optional[np.ndarray] = None,
) -> float:
    mode = str(spec.get("operating_cost_mode", "per_component_linear"))
    if mode == "per_component_linear":
        intercept = float(spec["operating_reward_intercept"])
        slope = float(spec["operating_reward_slope"])
        costs = intercept + slope * component_states.astype(np.float32)
        if bool(spec.get("operating_cost_action0_only", False)):
            if actions is None:
                raise ValueError("actions are required for action-dependent operating cost")
            costs = costs[np.asarray(actions, dtype=np.int32) == 0]
        return float(np.sum(costs))

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
    if mode == "remaining_life":
        remaining_life = np.asarray(spec["expected_remaining_life"], dtype=np.float64)
        return (
            float(spec["cpm_base"])
            + float(spec["cpm_rul_rate"]) * remaining_life[states_arr]
        )
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
    operating_cost = _operating_cost(component_states, spec, actions_arr)
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
    transition_dtype = (
        np.float64
        if bool(spec.get("vectorized_component_transitions", False))
        else np.float32
    )
    transition_no_maintenance = np.asarray(
        spec["transition_no_maintenance"], dtype=transition_dtype
    )
    component_transitions = component_transition_matrices(
        component_states.shape[0], env_name=env_name
    )

    if bool(spec.get("vectorized_component_transitions", False)):
        if component_transitions is None:
            transition_rows = transition_no_maintenance[component_states]
        else:
            component_indices = np.arange(component_states.shape[0])
            transition_rows = component_transitions[
                component_indices, component_states
            ]
        state_cdf = np.cumsum(transition_rows, axis=1)
        state_cdf[:, -1] = 1.0
        uniforms = rng.random_sample(component_states.shape[0])
        no_maintenance_next = np.sum(
            uniforms[:, np.newaxis] > state_cdf, axis=1
        ).astype(np.int32)
        next_states = np.where(actions_arr == 1, 0, no_maintenance_next)
    else:
        next_states = []
        for component_index, (state, action) in enumerate(
            zip(component_states, actions_arr)
        ):
            if action == 1:
                next_states.append(0)
            else:
                transition = (
                    transition_no_maintenance
                    if component_transitions is None
                    else component_transitions[component_index]
                )
                next_states.append(
                    rng.choice(state_values, p=transition[state])
                )
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


def opportunity_threshold_policy(
    states: Sequence[int],
    trigger: Optional[int] = None,
    opportunity_threshold: Optional[int] = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    trigger_value = int(
        spec.get("opportunity_trigger", spec["threshold"])
        if trigger is None else trigger
    )
    opportunity_value = int(
        spec.get("opportunity_threshold", trigger_value)
        if opportunity_threshold is None else opportunity_threshold
    )
    if opportunity_value > trigger_value:
        raise ValueError("opportunity_threshold must not exceed trigger")
    component_states, _ = split_observation(states, env_name=env_name)
    has_trigger = bool(np.any(component_states >= trigger_value))
    actions = (
        (component_states >= opportunity_value).astype(np.int32)
        if has_trigger else np.zeros(component_states.shape[0], dtype=np.int32)
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
    corrected_arr = np.clip(actions_arr, 0, 1).astype(np.int32)
    corrected_arr[component_states == failed_state] = 1
    if enforce_zero:
        corrected_arr[component_states == 0] = 0

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
        component_transition_matrices(num_machines, env_name=env_name)
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
