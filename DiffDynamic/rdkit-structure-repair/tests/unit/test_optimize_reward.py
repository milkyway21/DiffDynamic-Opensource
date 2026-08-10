"""Unit tests: the reward function's sign conventions.

Every term is a "better minus worse" difference, and getting one sign backwards
would quietly steer the whole optimizer the wrong way, so each is pinned
individually.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from structure_repair.optimize.reward import (
    compute_properties,
    compute_reward,
    count_structural_alerts,
    weights_from_config,
)

BASE = {
    "qed": 0.50,
    "sa": 4.0,
    "rotb": 6.0,
    "alerts": 1.0,
    "tpsa": 90.0,
    "mw": 350.0,
    "heavy_atoms": 25.0,
}


def shifted(**changes):
    out = dict(BASE)
    out.update(changes)
    return out


def total(after, **kwargs):
    return compute_reward(BASE, after, **kwargs)["total"]


def test_no_change_is_worth_nothing():
    assert compute_reward(BASE, dict(BASE))["total"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "term,better,worse",
    [
        ("qed", {"qed": 0.70}, {"qed": 0.30}),
        ("sa", {"sa": 2.5}, {"sa": 6.0}),
        ("rotb", {"rotb": 3.0}, {"rotb": 9.0}),
        ("alerts", {"alerts": 0.0}, {"alerts": 3.0}),
    ],
)
def test_each_term_points_the_right_way(term, better, worse):
    assert total(shifted(**better)) > 0
    assert total(shifted(**worse)) < 0


def test_tpsa_is_rewarded_towards_the_window_not_upwards():
    """TPSA is a window, not a direction: 300 is as wrong as 5."""
    outside_low = compute_reward(shifted(tpsa=5.0), shifted(tpsa=60.0))
    outside_high = compute_reward(shifted(tpsa=300.0), shifted(tpsa=120.0))
    leaving = compute_reward(shifted(tpsa=90.0), shifted(tpsa=300.0))
    assert outside_low["tpsa"] > 0
    assert outside_high["tpsa"] > 0
    assert leaving["tpsa"] < 0


def test_cost_is_subtracted():
    assert total(dict(BASE), transform_cost=3.0) < 0


def test_bonus_is_added():
    assert total(dict(BASE), transform_bonus=0.25) == pytest.approx(0.25, abs=1e-6)


def test_breakdown_sums_to_total():
    breakdown = compute_reward(
        BASE,
        shifted(qed=0.62, sa=3.2, rotb=4.0, alerts=0.0, tpsa=110.0, heavy_atoms=27.0),
        transform_cost=1.5,
        transform_bonus=0.1,
    )
    parts = sum(v for k, v in breakdown.items() if k != "total")
    assert parts == pytest.approx(breakdown["total"], abs=1e-4)


def test_weights_can_switch_a_term_off():
    only_qed = weights_from_config(
        {"reward": {"w_sa": 0.0, "w_rotb": 0.0, "w_alert": 0.0, "w_tpsa": 0.0, "w_le": 0.0}}
    )
    breakdown = compute_reward(BASE, shifted(sa=1.0, rotb=0.0, alerts=0.0), weights=only_qed)
    assert breakdown["total"] == pytest.approx(0.0, abs=1e-9)


def test_ligand_efficiency_uses_docking_scores_when_available():
    """With real Vina numbers the LE term must follow affinity per heavy atom."""
    improved = compute_reward(
        BASE, shifted(heavy_atoms=26.0), vina_before=-7.0, vina_after=-9.0
    )
    diluted = compute_reward(
        BASE, shifted(heavy_atoms=40.0), vina_before=-7.0, vina_after=-7.1
    )
    assert improved["le"] > 0
    assert diluted["le"] < 0


def test_growing_without_gain_is_penalised_when_no_docking_score_exists():
    assert compute_reward(BASE, shifted(heavy_atoms=32.0))["le"] < 0


# ------------------------------------------------------------------ properties


def test_compute_properties_covers_every_reward_input():
    props = compute_properties(Chem.MolFromSmiles("O=C(O)Cc1ccc(OC)cc1"))
    for key in ("qed", "sa", "rotb", "tpsa", "mw", "logp", "alerts", "heavy_atoms"):
        assert key in props
    assert props["heavy_atoms"] == 12
    assert 0.0 <= props["qed"] <= 1.0
    assert 1.0 <= props["sa"] <= 10.0


def test_structural_alerts_are_detected():
    clean = count_structural_alerts(Chem.MolFromSmiles("CNC(=O)c1ccccc1"))
    nitro = count_structural_alerts(Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1"))
    assert nitro > clean
