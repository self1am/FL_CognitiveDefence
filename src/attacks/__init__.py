"""
Attack implementations for federated learning.

This module provides various attack strategies including:
- Static attacks: label_flip, gradient_noise
- Adaptive attacks: stat_opt, dny_opt, min_max, min_sum
"""

from .base_attack import BaseAttack
from .label_flip import LabelFlipAttack
from .gradient_noise import GradientNoiseAttack
from .adaptive_base import AdaptiveAttack
from .stat_opt_attack import StatOptAttack
from .dny_opt_attack import DnyOptAttack
from .min_max_attack import MinMaxAttack
from .min_sum_attack import MinSumAttack

__all__ = [
    'BaseAttack',
    'LabelFlipAttack',
    'GradientNoiseAttack',
    'AdaptiveAttack',
    'StatOptAttack',
    'DnyOptAttack',
    'MinMaxAttack',
    'MinSumAttack',
]
