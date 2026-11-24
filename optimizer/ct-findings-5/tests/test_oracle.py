"""The oracle-examples matching (oracle_match.py, kb_oracle.py; spec planning/vista_eval_vlm/2026-11-24-oracle-matched-
examples.md). Synthetic items only; expected values worked out by hand from the rules.

    cd optimizer/ct-findings-5 && uv run -q --no-project --with pytest python -m pytest -q tests/test_oracle.py
"""

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import oracle_match as om  # noqa: E402


def test_sites():
    assert om.site("lung_solid", "right upper lobe apical segment") == "RUL"
    assert om.site("lung_ground_glass", "lingula") == "LUL"
    assert om.site("lung_solid", "peripheral", "left") == "left-lung"
    assert om.site("lymph_node", "right lower paratracheal (4R)") == "paratracheal"
    assert om.site("lymph_node", "subcarinal station 7") == "subcarinal"
    assert om.site("lymph_node", "left axillary") == "axillary"
    assert om.site("lymph_node", "retroperitoneal") == "other"
    assert om.site("liver", "segment 7") == "right"
    assert om.site("liver", "left lateral segment") == "left"
    assert om.side_of_site("RLL") == "right" and om.side_of_site("left-lung") == "left"


def item(i, role, person, loc="", side=None, matched=None, track="lung_solid", b="5-9"):
    d = {"item": i, "role": role, "person": person, "track": track, "bin": b, "location": loc, "side": side}
    if matched:
        d["matched_to"] = matched
    return d


def test_perception_sets_are_balanced_and_from_other_patients():
    items = [item("p1", "positive", "A", "right upper lobe"), item("p2", "positive", "B", "right upper lobe"),
             item("p3", "positive", "C", "left lower lobe"), item("p4", "positive", "D", "left lower lobe"),
             item("n1", "negative", "E", matched="p1"), item("n2", "negative", "F", matched="p3"),
             item("n3", "negative", "G", matched="p2"), item("n4", "negative", "H", matched="p4")]
    sets, close = om.perception_sets(items, seed=1)
    assert set(sets) == {"p1", "p2", "p3", "p4", "n1", "n2", "n3", "n4"}
    persons = {i["item"]: i["person"] for i in items}
    for case, exs in sets.items():
        assert sorted(e["kind"] for e in exs) == ["decoy", "no", "yes"]  # the same structure for every case
        assert all(e["person"] != persons[case] for e in exs)  # never the case's own patient
        kinds = {e["kind"]: e for e in exs}
        assert kinds["yes"]["site"] == kinds["no"]["site"] != kinds["decoy"]["site"]
    # a negative takes its matched positive's site: n1 (matched to p1, RUL) gets RUL yes and no
    k = {e["kind"]: e for e in sets["n1"]}
    assert k["yes"]["site"] == "RUL" and k["no"]["site"] == "RUL" and k["decoy"]["site"] == "LLL"
    assert close["yes exact"] == 8 and close["no exact"] == 8


def test_perception_falls_back_to_the_same_side_and_counts_it():
    items = [item("p1", "positive", "A", "right upper lobe"), item("p2", "positive", "B", "right lower lobe"),
             item("p3", "positive", "C", "left lower lobe"), item("n1", "negative", "E", matched="p2"),
             item("n2", "negative", "F", matched="p3")]
    sets, close = om.perception_sets(items, seed=1)
    k = {e["kind"]: e for e in sets["p1"]}
    assert k["yes"]["site"] == "RLL"  # no other RUL lesion: same side
    assert close["yes same side"] >= 1


def test_kb_oracle_helpers():
    import importlib
    kbo = importlib.import_module("kb_oracle")
    assert kbo.node_bin(7) == "5-9" and kbo.node_bin(12) == "10-19" and kbo.node_bin(25) == ">=20" and kbo.node_bin(None) is None
    # the small-finding rule: a node's short axis when the extraction says the size is long x short, else the largest
    assert kbo.node_mm({"size_current": "1.4 x 0.9 cm", "size_axis": "long_x_short"}) == 9.0
    assert kbo.node_mm({"size_current": "1.4 x 0.9 cm", "size_axis": ""}) == 14.0
    assert kbo.node_mm({"size_current": "up to 6 mm"}) is None
    assert kbo.entry_lobes({"title": "Bilateral ground glass", "words": []}) == {"RUL", "RML", "RLL", "LUL", "LLL"}
    assert kbo.entry_lobes({"title": "Right upper lobe consolidation post RT", "words": ["rul"]}) == {"RUL"}
    assert kbo.entry_lobes({"title": "Lingular band", "words": []}) == {"LUL"}
    assert Counter(kbo.TARGETS) == Counter(("lesion_lymph_nodes", "pneumonitis"))
