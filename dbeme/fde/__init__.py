"""Open-source FDE backend for the dataset-based EME solver."""

from .base import FDEBackend, ModeData
from .cross_section import BiLevelPair, BiLevelStrip, CoupledStrips, CrossSection, FullEtchStrip
from .emepy_fde import EmepyFDE
from .materials import (
    ConstantIndex,
    Material,
    PyOptikMaterial,
    PyOptikUnavailable,
    SellmeierMaterial,
    air,
    as_material,
    gold, silica,
    silicon,
    silicon_nitride,
)

__all__ = [
    "FDEBackend",
    "ModeData",
    "CrossSection",
    "FullEtchStrip",
    "CoupledStrips",
    "BiLevelStrip",
    "BiLevelPair",
    "EmepyFDE",
    "Material",
    "ConstantIndex",
    "SellmeierMaterial",
    "PyOptikMaterial",
    "PyOptikUnavailable",
    "as_material",
    "silicon",
    "silica",
    "silicon_nitride",
    "air",
]
