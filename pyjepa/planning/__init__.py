"""Sampling-based planners (CEM, MPPI, Random Shooting) over a learned world model.

These planners treat the JEPA world model as a black-box cost function::

    cost(action_sequence)  →  scalar

and search for the action sequence that minimizes it. They work for any model
that implements ``get_cost(info, action_candidates)`` — including LeWM.
"""

from .cem import CEMPlanner, CEMConfig
from .mppi import MPPIPlanner, MPPIConfig
from .policy import PlanConfig, RandomPolicy, WorldModelPolicy
from .random_shoot import RandomShootingPlanner

__all__ = [
    "CEMConfig",
    "CEMPlanner",
    "MPPIConfig",
    "MPPIPlanner",
    "PlanConfig",
    "RandomPolicy",
    "RandomShootingPlanner",
    "WorldModelPolicy",
]
