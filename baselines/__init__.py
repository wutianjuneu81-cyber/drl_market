# Baselines module
from .controllers.rule_based import RuleBasedController

try:
    from .controllers.mpc import MPCController
except ImportError:
    MPCController = None

__all__ = ["RuleBasedController", "SingleLayerRLController", "MPCController", "evaluate_baseline"]