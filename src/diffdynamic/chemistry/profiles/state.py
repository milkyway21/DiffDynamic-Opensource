from __future__ import annotations
from diffdynamic.io.paths import project_root

"""Shared imports, constants and process configuration for diffdynamic.chemistry.profiles."""

import json


import re


from collections import Counter, deque


from pathlib import Path


from typing import Any, Iterable, Optional


from rdkit import Chem, RDLogger


from rdkit.Chem import BRICS


RDLogger.DisableLog("rdApp.*")


PROFILE_VERSION = 4


SUPPORTED_PROFILE_VERSIONS = {1, 2, 3, PROFILE_VERSION}


SUPPORTED_GENERATION_ELEMENTS = {
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
}
