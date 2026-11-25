"""The library arm's local caches (spec planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-library.md), on the work
folder's disk so sessions get hard links, not copies.

    train   config.TRAIN_SCANS_DIR/<id>/{ct/ct.npz, ct/meta.json, organs.json, organ_mask.npz, answer.json, report.txt}
            for every discovery TRAIN scan (data discovery/scans.jsonl, 599), index.json (id, cause, opaque patient
            group), README.md, and _tools/{similar.py, embeddings.npz} (MerlinOnc cosine over TRAIN only). TRAIN only:
            a VAL id stops it. Every file read-only. Resumes.
    lit     config.LIT_ASSETS_DIR/lit/<id>.<ext> for both literature image sets and lit/index.json (id, file, cause,
            pattern, caption, panel, source, licence, use), ids "oa-..." (open access) and "mac-..." (paywalled papers).
    check   $0: every discovery scan staged with all its files; every literature image listed and present. PASS/FAIL.

    cd optimizer/pneumonitis-variants
    LIBRARY=1 uv run -q --project $AGENTIC_LABEL_OPT --with numpy python -B scripts/library_assets.py train|lit|check

Prints counts only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("LIBRARY", "1")
import config  # noqa: E402

SIMILAR = '''"""Scans most like a given scan (MerlinOnc whole-scan embedding, cosine), among the TRAIN scans in this folder.

    python3 -B similar.py SCAN_ID [--cause radiation] [--limit 10]
"""
import argparse, json
from pathlib import Path
import numpy as np
here = Path(__file__).resolve().parent
z = np.load(here / "embeddings.npz"); ids = [str(i) for i in z["ids"]]; e = z["emb"].astype("float64")
e /= np.linalg.norm(e, axis=1, keepdims=True)
index = {r["id"]: r for r in json.loads((here.parent / "index.json").read_text())}
ap = argparse.ArgumentParser(); ap.add_argument("scan"); ap.add_argument("--cause"); ap.add_argument("--limit", type=int, default=10)
a = ap.parse_args()
i = ids.index(a.scan); sims = e @ e[i]
order = [j for j in np.argsort(-sims) if ids[j] != a.scan and index[ids[j]]["group"] != index[a.scan]["group"]]
order = [j for j in order if not a.cause or index[ids[j]]["cause"] == a.cause][: a.limit]
for j in order:
    print(f"{ids[j]}  cause={index[ids[j]]['cause']}  similarity={sims[j]:.3f}")
'''


def discovery_rows() -> list[dict]:
    return [json.loads(x) for x in (config.DATA_DIR / "discovery" / "scans.jsonl").read_text().splitlines() if x]


def cmd_train(_args) -> int:
    import numpy as np

    rows = discovery_rows()
    val = set(json.loads((config.DATA_DIR / "pools.json").read_text())["val"])
    if {r["id"] for r in rows} & val:
        raise SystemExit("a VAL scan is in the discovery set")
    root = config.TRAIN_SCANS_DIR
    root.mkdir(parents=True, exist_ok=True)
    c = Counter()
    for r in rows:
        d, i = root / r["id"], r["id"]
        if (d / "answer.json").is_file():
            c["kept"] += 1
            continue
        tmp = root / f".{i}.part"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "ct").mkdir(parents=True)
        for name in ("ct.npz", "meta.json"):
            shutil.copy2(config.DATA_DIR / "patients" / i / name, tmp / "ct" / name)
        shutil.copy2(config.DATA_DIR / "organs" / f"{i}.json", tmp / "organs.json")
        shutil.copy2(config.DATA_DIR / "masks" / f"{i}.npz", tmp / "organ_mask.npz")
        rep = config.DATA_DIR / "train_reports" / f"{i}.txt"
        if rep.is_file():
            shutil.copy2(rep, tmp / "report.txt")
        else:
            c["no_report"] += 1
        (tmp / "answer.json").write_text(json.dumps({"cause": r["cause"]}))
        for f in tmp.rglob("*"):
            if f.is_file():
                f.chmod(0o444)
        tmp.rename(d)
        c["staged"] += 1
    (root / "index.json").write_text(json.dumps([{"id": r["id"], "cause": r["cause"], "group": r["group"]} for r in rows]))
    tools = root / "_tools"
    tools.mkdir(exist_ok=True)
    (tools / "similar.py").write_text(SIMILAR)
    with np.load(config.DATA_DIR / "embeddings.npz") as z:
        idx = {str(k): n for n, k in enumerate(z["ids"])}
        keep = [r["id"] for r in rows]
        np.savez_compressed(tools / "embeddings.npz", ids=np.array(keep), emb=z["emb"][[idx[k] for k in keep]])
    (root / "README.md").write_text(
        "# The TRAIN scans\n\nOne folder per scan: ct/ (ct.npz, meta.json), organs.json, organ_mask.npz, answer.json "
        "(the cause: the answer key), report.txt (the radiology report the answer was read from). index.json lists every "
        "scan with its cause and an opaque patient group (scans of one patient share it). Search the reports with Grep "
        "(e.g. Grep for 'osimertinib' in */report.txt). Scans most like a given one: "
        "`python3 -B _tools/similar.py <id> [--cause C]`.\n")
    print(f"discovery TRAIN scans {len(rows)}: {dict(c)}; by cause {dict(Counter(r['cause'] for r in rows))}")
    return 0


def lit_records() -> list[tuple[Path, dict, str]]:
    out = []
    for base, prefix in zip(config.LIT_IMAGE_SETS, ("oa", "mac")):
        m = json.loads((base / "manifest.json").read_text())
        for r in m if isinstance(m, list) else m.get("figures", m):
            out.append((base, r, prefix))
    return out


def cmd_lit(_args) -> int:
    root = config.LIT_ASSETS_DIR / "lit"
    root.mkdir(parents=True, exist_ok=True)
    index, seen = [], set()
    for base, r, prefix in lit_records():
        src = base / r["file"]
        lid = f"{prefix}-{Path(r['file']).stem}".lower().replace("_", "-")
        if lid in seen:
            raise SystemExit("two literature images have one id")
        seen.add(lid)
        dest = root / f"{lid}{src.suffix.lower()}"
        if not dest.is_file():
            shutil.copy2(src, dest)
            dest.chmod(0o444)
        s = r.get("source", {})
        index.append({"id": lid, "file": dest.name, "cause": r.get("cause"), "pattern": r.get("pattern"),
                      "caption": r.get("caption"), "panel": r.get("panel"),
                      "source": {k: s.get(k) for k in ("title", "journal", "site or journal", "year", "url", "pmid", "pmcid") if s.get(k)},
                      "licence": r.get("licence"), "use": r.get("use", "may be shown to the reader")})
    f = root / "index.json"
    if f.exists():
        f.chmod(0o644)
    f.write_text(json.dumps(index, indent=1))
    f.chmod(0o444)
    print(f"literature images {len(index)}: by cause {dict(Counter(r['cause'] for r in index))}; "
          f"by set {dict(Counter(r['id'].split('-')[0] for r in index))}")
    return 0


def cmd_check(_args) -> int:
    ok, rows = True, discovery_rows()
    miss = [r["id"] for r in rows if not all((config.TRAIN_SCANS_DIR / r["id"] / n).is_file()
                                              for n in ("ct/ct.npz", "ct/meta.json", "organs.json", "organ_mask.npz", "answer.json"))]
    ok &= not miss
    print(f"TRAIN scans staged {len(rows) - len(miss)}/{len(rows)}")
    idx = json.loads((config.LIT_ASSETS_DIR / "lit" / "index.json").read_text())
    absent = [r for r in idx if not (config.LIT_ASSETS_DIR / "lit" / r["file"]).is_file()]
    ok &= not absent and len(idx) == len(lit_records())
    print(f"literature images {len(idx) - len(absent)}/{len(lit_records())}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("train", "lit", "check"):
        sub.add_parser(n)
    args = ap.parse_args()
    return {"train": cmd_train, "lit": cmd_lit, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
