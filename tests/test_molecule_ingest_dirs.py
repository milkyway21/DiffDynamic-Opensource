"""Nested extract/eval directories are scanned for molecule ingest."""
import os

from server.molecule_ingest import (
    _eval_dirs_to_scan,
    _read_sdf_props,
    _usable_smiles,
    match_sdf_for_molecule,
)
from server.api import _decode_user_path


def test_eval_dirs_to_scan_nested_extract(tmp_path):
    root = tmp_path / "extract_extract_0002"
    nested = root / "eval_20260904_195345_0_gfquadratic"
    nested.mkdir(parents=True)
    (nested / "eval_results_20260904_195345_final_x.pt").write_bytes(b"stub")
    (root / "readme.txt").write_text("no pts here")
    dirs = _eval_dirs_to_scan(str(root))
    assert len(dirs) == 1
    assert os.path.samefile(dirs[0], nested)


def test_eval_dirs_to_scan_flat_eval(tmp_path):
    eval_dir = tmp_path / "eval_job"
    eval_dir.mkdir()
    (eval_dir / "eval_results_1.pt").write_bytes(b"stub")
    dirs = _eval_dirs_to_scan(str(eval_dir))
    assert len(dirs) == 1
    assert os.path.samefile(dirs[0], eval_dir)


def test_usable_smiles_skips_empty():
    assert _usable_smiles(None) is None
    assert _usable_smiles("") is None
    assert _usable_smiles("None") is None
    assert _usable_smiles("  CCO  ") == "CCO"


def _write_sdf(path, smiles, index):
    path.write_text(
        "\n"
        "     RDKit          3D\n\n"
        "  0  0  0  0  0  0  0  0  0  0999 V2000\n"
        "M  END\n"
        f">  <SMILES>  (1) \n{smiles}\n\n"
        f">  <Molecule_Index>  (1) \n{index}\n\n"
        "$$$$\n"
    )


def test_read_sdf_props_rdkit_count_suffix(tmp_path):
    p = tmp_path / "mol.sdf"
    _write_sdf(p, "CCO", 4)
    rec = _read_sdf_props(str(p))
    assert rec["smiles"] == "CCO"
    assert rec["index"] == 4


def test_match_sdf_by_index_not_glob_order(tmp_path):
    recon = tmp_path / "reconstructed_molecules"
    recon.mkdir()
    _write_sdf(recon / "zzz_good.sdf", "CCO", 0)
    _write_sdf(recon / "aaa_other.sdf", "CCC", 1)
    recs = [_read_sdf_props(str(p)) for p in sorted(recon.glob("*.sdf"))]
    used = set()
    path0 = match_sdf_for_molecule(0, "CCO", recs, used)
    used.add(path0)
    path1 = match_sdf_for_molecule(1, "CCC", recs, used)
    assert path0.endswith("zzz_good.sdf")
    assert path1.endswith("aaa_other.sdf")
    assert match_sdf_for_molecule(0, "CCO", recs, used) is None


def test_skip_failed_row_does_not_steal_sdf(tmp_path):
    recon = tmp_path / "reconstructed_molecules"
    recon.mkdir()
    _write_sdf(recon / "only.sdf", "CCO", 2)
    recs = [_read_sdf_props(str(recon / "only.sdf"))]
    used = set()
    assert match_sdf_for_molecule(0, None, recs, used) is None
    path = match_sdf_for_molecule(2, "CCO", recs, used)
    assert path.endswith("only.sdf")


def test_decode_user_path_unquotes_slash():
    assert _decode_user_path("%2Fdata%2Fye%2FDiffDynamic%2Ffoo.sdf") == "/data/ye/DiffDynamic/foo.sdf"
    assert _decode_user_path("") is None
    assert _decode_user_path(None) is None
