"""Report-matched (oracle) example sets (spec planning/vista_eval_vlm/2026-11-24-oracle-matched-examples.md, Approved
2026-11-24). An oracle picks each case's examples from what its own report says, so it is an upper bound: it tests
whether comparable examples help, and cannot be used at test time.

Every case gets the same structure, whether its answer is yes or no, so the set never reveals the answer:
    yes     a confirmed finding of the same kind and size bin at the case's site, from another patient
    no      the same site in another patient whose report says it is clear
    decoy   a confirmed finding of the same kind and size bin at a different site, from another patient
A case's site is where its report puts the finding; a clear case takes the site of the lesion it was matched to (the
perception study's negatives are each matched to a positive at the same organ level), so clear and positive cases
carry the same distribution of sites. When no example exists at the exact site, the nearest site is used (same side,
then any) and the fallback is counted.

Sites: lung = lobe (RUL, RML, RLL, LUL, LLL; side when the lobe is not stated); liver = right / left / caudate lobe;
lymph node = station group (paratracheal, subcarinal, hilar, axillary, supraclavicular, other).
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

LOBES = (("RUL", r"right upper|\brul\b|right (lung )?ap(ex|ical)"), ("RML", r"right middle|\brml\b|middle lobe"),
         ("RLL", r"right lower|\brll\b|right (lung )?base"), ("LUL", r"left upper|\blul\b|lingul|left (lung )?ap(ex|ical)"),
         ("LLL", r"left lower|\blll\b|left (lung )?base"))
STATIONS = (("supraclavicular", r"supraclav|lower cervical"), ("axillary", r"axill"), ("subcarinal", r"subcarin"),
            ("hilar", r"hil(ar|um)|interlobar|bronchopulm"),
            ("paratracheal", r"paratrache|pretrache|prevascular|aortopulmonary|\bap window|precarinal|mediastin"))
LIVER = (("caudate", r"caudate"), ("left", r"left (hepatic )?lobe|segment (2|3|4)|\bseg(ment)? ?(ii|iii|iv)\b|lateral segment"),
         ("right", r"right (hepatic )?lobe|segment (5|6|7|8)|\bseg(ment)? ?(v|vi|vii|viii)\b|dome"))
SIDE_OF_LOBE = {"RUL": "right", "RML": "right", "RLL": "right", "LUL": "left", "LLL": "left"}


def _first(table, text: str):
    t = (text or "").lower()
    for name, rx in table:
        if re.search(rx, t):
            return name
    return None


def site(track: str, location: str, side: str | None = None) -> str:
    """The site of one finding from its location text (and side, for lung lesions with no lobe)."""
    if track.startswith("lung"):
        return _first(LOBES, location) or (f"{side}-lung" if side in ("right", "left") else "lung")
    if track == "liver":
        return _first(LIVER, location) or "liver"
    return _first(STATIONS, location) or "other"


def side_of_site(s: str) -> str | None:
    if s in SIDE_OF_LOBE:
        return SIDE_OF_LOBE[s]
    return s.split("-")[0] if s.endswith("-lung") else None


def _nearest(candidates: list[dict], want: str, key) -> tuple[dict | None, str]:
    """A candidate at the exact site, else on the same side, else any; with how close it was."""
    exact = [c for c in candidates if key(c) == want]
    if exact:
        return exact[0], "exact"
    sd = side_of_site(want)
    same_side = [c for c in candidates if sd and side_of_site(key(c)) == sd]
    if same_side:
        return same_side[0], "same side"
    return (candidates[0], "any") if candidates else (None, "none")


def perception_sets(items: list[dict], seed: int, bins: tuple[str, ...] | None = None):
    """For each positive and negative of the perception study (small_finding/items.py), its oracle set as three item
    dicts with a "kind" (yes / no / decoy), shuffled per case; and counts of how close each pick was. References come
    from other patients only: positives and layer-2 examples for yes / decoy, negatives for no."""
    by_id = {i["item"]: i for i in items}
    def s_of(i):
        src = by_id[i["matched_to"]] if i["role"] == "negative" else i
        return site(i["track"], src.get("location", ""), src.get("side"))

    lesions, clear = defaultdict(list), defaultdict(list)
    for i in items:
        cell = (i["track"], i["bin"])
        if i["role"] in ("positive", "example"):
            lesions[cell].append(i)
        elif i["role"] == "negative":
            clear[cell].append(i)
    sets, close = {}, Counter()
    for t in (i for i in items if i["role"] in ("positive", "negative")):
        if bins and t["bin"] not in bins:
            continue
        cell, want = (t["track"], t["bin"]), s_of(t)
        rng = random.Random(f"{seed}:oracle:{t['item']}")
        others = [x for x in lesions[cell] if x["person"] != t["person"]]
        rng.shuffle(others)
        yes, how_yes = _nearest(others, want, s_of)
        decoys = [x for x in others if x is not yes and s_of(x) != (s_of(yes) if yes else want)]
        decoy = decoys[0] if decoys else None
        nos = [x for x in clear[cell] if x["person"] != t["person"] and x["item"] != t["item"]]
        rng.shuffle(nos)
        no, how_no = _nearest(nos, want, s_of)
        if yes is None or no is None or decoy is None:
            close["incomplete"] += 1
            continue
        close[f"yes {how_yes}"] += 1
        close[f"no {how_no}"] += 1
        exs = [{**yes, "kind": "yes", "site": s_of(yes)}, {**no, "kind": "no", "site": s_of(no)},
               {**decoy, "kind": "decoy", "site": s_of(decoy)}]
        rng.shuffle(exs)
        sets[t["item"]] = exs
    return sets, close
