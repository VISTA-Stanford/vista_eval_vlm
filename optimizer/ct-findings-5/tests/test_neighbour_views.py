"""The neighbour-CT view counter (scripts/retrieval_usage.py neighbour_views): the "opened a neighbour CT" share and
the runner's half-of-sessions stop both rest on it. Synthetic shell commands only; expected ranks worked out by hand
from the counter's rules (each shell piece alone, cd followed, --help/--info and the session's own ct/ never count,
a loop or glob counts each rank it covers once).

    cd optimizer/ct-findings-5 && uv run -q --project $AGENTIC_LABEL_OPT --with pytest python -m pytest -q tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retrieval_usage import neighbour_views  # noqa: E402

V = "python3 -B tools/view_ct.py"

CASES = [
    # the shape seen in a real arm A session: help, README and a loop over all five neighbours
    (f"cd /workspace/program; {V} --help; cat ../call/neighbours/README.md", []),
    (f"cd /workspace/program; for i in 1 2 3 4 5; do {V} ../call/neighbours/n$i/ct --plane axial --at 0.6; done",
     ["1", "2", "3", "4", "5"]),
    (f"for k in 2 4; do {V} ../call/neighbours/n$k/ct; done", ["2", "4"]),
    (f"{V} ../call/neighbours/n{{1..5}}/ct --count 4", ["1", "2", "3", "4", "5"]),
    # the session's own scan, and calls that draw nothing
    (f"{V} ../call/ct --plane axial --count 12 --window lung", []),
    (f"{V} ../call/neighbours/n2/ct --info", []),
    (f"{V} ../call/neighbours/n1/ct --help", []),
    (f"{V} --help && ls ../call/neighbours/n3", []),
    # direct paths: two views of one neighbour
    (f"{V} ../call/neighbours/n4/ct --at 0.5; {V} ../call/neighbours/n4/ct --at 0.7", ["4", "4"]),
    # cd into a neighbour, or into neighbours/ then a relative n<k>/
    ("cd /workspace/call/neighbours/n3 && python3 /workspace/program/tools/view_ct.py ct --plane coronal", ["3"]),
    ("cd ../call/neighbours; python3 ../../program/tools/view_ct.py n1/ct", ["1"]),
    # leaving neighbours/ again: the own ct/ does not count
    ("cd /workspace/call/neighbours/n5; cd /workspace/call; python3 /workspace/program/tools/view_ct.py ct", []),
    # reading answers or reports is not a view
    ("cat ../call/neighbours/n1/answers.json ../call/neighbours/n1/report.txt", []),
]


@pytest.mark.parametrize("command,expected", CASES)
def test_neighbour_views(command, expected):
    assert sorted(neighbour_views(command)) == sorted(expected)
