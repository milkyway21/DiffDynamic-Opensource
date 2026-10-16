"""Portable regression tests for frozen TargetDiff reconstruction."""

import hashlib
from pathlib import Path

import pytest
from rdkit import Chem

from evaluate_pt_with_correct_reconstruct import reconstruct_molecule
from utils import reconstruct_targetdiff_legacy


UPSTREAM_SHA256 = '2d89a69cb9cae97e7d1e14b7c7e097c6d7aa25efb351c649c84893ec8f755d9b'


def test_frozen_original_targetdiff_source():
    path = Path(reconstruct_targetdiff_legacy.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == UPSTREAM_SHA256


def test_legacy_mode_rejects_scaffold_constraints():
    with pytest.raises(ValueError, match='de novo coordinates only'):
        reconstruct_molecule([[0, 0, 0]], [1], n_scaffold=1,
                             reconstruction_method='targetdiff_legacy')


def test_legacy_reconstruction_exception_is_accounted_for(monkeypatch):
    def fail(*args, **kwargs):
        raise reconstruct_targetdiff_legacy.MolReconsError('fixture failure')

    monkeypatch.setattr(reconstruct_targetdiff_legacy, 'reconstruct_from_generated', fail)
    mol, error = reconstruct_molecule([[0, 0, 0]], [1],
                                      reconstruction_method='targetdiff_legacy')
    assert mol is None and error['error_type'] == 'MolReconsError'
