"""Loss functions and regularizers for JEPA training.

- :class:`JEPALoss` — the canonical V-JEPA prediction loss + variance regularizer.
- :class:`SIGReg` — LeWM's Sketched Isotropic Gaussian Regularizer.
- :class:`VICRegLoss` — variance/invariance/covariance regularizer baseline.
- :func:`prediction_loss` — bare L_p / smooth-L1 prediction loss.
"""

from .jepa import JEPALoss, VarianceRegularizer
from .prediction import prediction_loss
from .sigreg import SIGReg
from .vicreg import VICRegLoss

__all__ = [
    "JEPALoss",
    "SIGReg",
    "VICRegLoss",
    "VarianceRegularizer",
    "prediction_loss",
]
