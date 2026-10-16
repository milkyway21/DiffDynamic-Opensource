from rdkit import Chem

from utils.visual_sim import fraggle_fragments, fraggle_similarity, prepare_query

LC409 = "COc1cccc2sc([*:1])cc12"
LC412 = "CNc1cccc2c1OCC([*:1])=C2"
THIOCHROMONE = "O=c1cc([*:1])sc2ccccc12"
CHROMENE = "C1=C([*:1])COc2ccccc21"
CHAIN = "CCCCCC[*:1]"


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


def test_self_similarity_near_one():
    for smiles in (LC409, LC412, "c1ccc2c([*:1])coc2c1"):
        query = prepare_query(_mol(smiles))
        score, match = fraggle_similarity(query, _mol(smiles))
        assert score >= 0.99, (smiles, score, match)
        assert fraggle_fragments(query)


def test_lc409_self_beats_thiochromone():
    query = prepare_query(_mol(LC409))
    self_score, _ = fraggle_similarity(query, _mol(LC409))
    thio, _ = fraggle_similarity(query, _mol(THIOCHROMONE))
    assert self_score > thio
    assert thio < 0.95


def test_lc412_chromene_beats_random_chain():
    query = prepare_query(_mol(LC412))
    chromene, _ = fraggle_similarity(query, _mol(CHROMENE))
    chain, _ = fraggle_similarity(query, _mol(CHAIN))
    assert chromene > chain
    assert chromene > 0.4
