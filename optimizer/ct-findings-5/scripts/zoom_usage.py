"""What the organ-zoom arms' sessions looked at (spec planning/vista_eval_vlm/2026-11-24-agent-organ-zoom.md), from
each session's view log (views.jsonl, written by tools/zoom_organ.py and tools/view_ct.py beside its ct/) and its
transcript (which images it opened with Read). Extends scripts/kb_usage.py's role for these arms; reuses
scripts/retrieval_usage.py's transcript readers.

    usage ANSWERS_JSON --out DIR [--train-check]

Per scan: zoomed images drawn and opened, per region (liver; lungs, any side or lobe; lymph nodes); grid images
drawn and opened; cropped grid views of each region (an opened axial view_ct image with --crop whose slices include
one in the region's slab range from organs.json; lymph nodes = bottom of the heart to top of the lungs); tool
refusals and crashes. The required look is met when each of the three regions has an opened close view: in the tool
arm a zoomed image, in the control arm a cropped grid view. With --train-check, also whether the session opened a
view within one slab of the cited lesion's slab (private train_check_items.jsonl): a zoomed image of the lesion's
region, or any axial grid image. Writes usage_per_scan.json (opaque ids, counts only) and usage_summary.json.

    cd optimizer/ct-findings-5
    uv run -q --project $AGENTIC_LABEL_OPT python -B scripts/zoom_usage.py usage ANSWERS_JSON --out DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import config  # noqa: E402
from retrieval_usage import events_file, tool_inputs  # noqa: E402

REGIONS = ("liver", "lungs", "lymph_nodes")
LUNG_ORGANS = {"lungs", "right_lung", "left_lung", "right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
               "left_upper_lobe", "left_lower_lobe"}
TRACK_REGION = {"lung_solid": "lungs", "lung_ground_glass": "lungs", "liver": "liver", "lymph_node": "lymph_nodes"}


def region_of(organ: str) -> str | None:
    return "liver" if organ == "liver" else "lymph_nodes" if organ == "lymph_nodes" else "lungs" if organ in LUNG_ORGANS else None


def region_ranges(organs: dict) -> dict[str, tuple[int, int]]:
    """Slab ranges of the three regions from an organs.json (lymph nodes: heart bottom to lungs top)."""
    o = {x["organ"]: (x["from_slice"], x["to_slice"]) for x in organs.get("organs", []) if x.get("present")}
    out = {k: o[k] for k in ("liver", "lungs") if k in o}
    if "lungs" in o:
        out["lymph_nodes"] = (o["heart"][0] if "heart" in o else o["lungs"][0], o["lungs"][1])
    return out


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def opened_paths(events: Path | None) -> set[str]:
    if events is None:
        return set()
    out = set()
    for name, text in tool_inputs(events):
        if name == "Read":
            out.add(json.loads(text).get("file_path", ""))
    return out


def count_one(log: list[dict], opened: set[str], ranges: dict, lesion: dict | None) -> dict:
    """One session's counts. An image counts as opened when its path was ever Read (a path drawn twice and read once
    counts both draws as opened: the log does not order draws against reads)."""
    out = {"zoom_drawn": 0, "zoom_opened": 0, "grid_drawn": 0, "grid_opened": 0, "refused": 0, "crashes": 0,
           **{f"zoom_opened_{r}": 0 for r in REGIONS}, **{f"crop_opened_{r}": 0 for r in REGIONS}}
    zoom_slabs: dict[str, set[int]] = {r: set() for r in REGIONS}
    grid_slabs: set[int] = set()
    for e in log:
        if "crash" in e:
            out["crashes"] += 1
            continue
        if "refused" in e:
            out["refused"] += 1
            continue
        is_open = e.get("out") in opened
        if e.get("tool") == "zoom_organ":
            out["zoom_drawn"] += 1
            r = region_of(e.get("organ", ""))
            if is_open and r:
                out["zoom_opened"] += 1
                out[f"zoom_opened_{r}"] += 1
                zoom_slabs[r].add(int(e["slab"]))
        elif e.get("tool") == "view_ct":
            out["grid_drawn"] += 1
            if not is_open:
                continue
            out["grid_opened"] += 1
            if e.get("plane") != "axial":
                continue
            grid_slabs.update(int(s) for s in e.get("slices", []))
            if e.get("crop"):
                for r, (lo, hi) in ranges.items():
                    if any(lo <= int(s) <= hi for s in e.get("slices", [])):
                        out[f"crop_opened_{r}"] += 1
    out["zoom_regions"] = sum(out[f"zoom_opened_{r}"] > 0 for r in REGIONS)
    out["crop_regions"] = sum(out[f"crop_opened_{r}"] > 0 for r in REGIONS)
    out["distinct_zoom_slabs"] = sum(len(v) for v in zoom_slabs.values())
    if lesion is not None:
        z, r = int(lesion["slab"]), TRACK_REGION[lesion["track"]]
        out["found_zoom"] = any(abs(s - z) <= 1 for s in zoom_slabs[r])
        out["found_any"] = out["found_zoom"] or any(abs(s - z) <= 1 for s in grid_slabs)
        out["track"], out["bin"] = lesion["track"], lesion["bin"]
    return out


def cmd_usage(args) -> int:
    answers_path = Path(args.answers)
    answers = json.loads(answers_path.read_text(encoding="utf-8"))["answers"]
    lesions = {}
    if args.train_check:
        lesions = {r["id"]: r for r in map(json.loads, (config.ZOOM_PRIVATE / "train_check_items.jsonl")
                                           .read_text(encoding="utf-8").splitlines()) if r}
    per, no_transcript, no_folder = {}, 0, 0
    for cid in sorted(answers):
        a = answers[cid]
        folder = Path(a["findings_log"]).parent if a.get("findings_log") else None
        if folder is None:
            no_folder += 1
        ev = events_file(answers_path.parent, cid)
        no_transcript += ev is None
        organs = json.loads((config.ZOOM_DIR / "organs" / f"{cid}.json").read_text(encoding="utf-8"))
        per[cid] = count_one(read_log(folder / "views.jsonl") if folder else [], opened_paths(ev),
                             region_ranges(organs), lesions.get(cid))
        per[cid]["error"] = a.get("error")
    n = len(per)
    if not n:
        print(f"FAIL: no answers in {answers_path}")
        return 1
    v = list(per.values())
    key = "zoom_regions" if config.ZOOM_ARM == "tool" else "crop_regions"
    summary = {
        "arm": config.ZOOM_ARM, "scans": n, "without_transcript": no_transcript, "without_session_folder": no_folder,
        "program_errors": sum(1 for x in v if x["error"]), "tool_crashes": sum(x["crashes"] for x in v),
        "refusals": sum(x["refused"] for x in v),
        "required_look_met": sum(1 for x in v if x[key] == 3), "required_look_missed": sum(1 for x in v if x[key] < 3),
        "sessions_with_a_zoom_image": sum(1 for x in v if x["zoom_drawn"]),
        **{f"median_{k}": statistics.median(x[k] for x in v)
           for k in ("zoom_drawn", "zoom_opened", "grid_drawn", "grid_opened", "distinct_zoom_slabs")},
        **{f"mean_{k}": round(statistics.mean(x[k] for x in v), 2)
           for k in ("zoom_drawn", "zoom_opened", "grid_drawn", "grid_opened", "distinct_zoom_slabs")},
        **{f"sessions_close_view_{r}": sum(1 for x in v if x[f"zoom_opened_{r}"] or x[f"crop_opened_{r}"])
           for r in REGIONS},
        "sessions_at_40_zoom_images": sum(1 for x in v if x["zoom_drawn"] >= 40),
    }
    if args.train_check:
        t = [x for x in v if "found_zoom" in x]
        summary["train_check"] = {
            "scans": len(t), "found_zoom": sum(x["found_zoom"] for x in t), "found_any_view": sum(x["found_any"] for x in t),
            "by_track_bin": {f"{tr} {b}": {"scans": len(c), "found_zoom": sum(x["found_zoom"] for x in c)}
                             for tr, b in sorted({(x["track"], x["bin"]) for x in t})
                             for c in [[x for x in t if (x["track"], x["bin"]) == (tr, b)]]}}
    summary["tool_broken"] = summary["tool_crashes"] > 0 or (config.ZOOM_ARM == "tool" and not summary["sessions_with_a_zoom_image"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "usage_per_scan.json").write_text(json.dumps(per, indent=0), encoding="utf-8")
    (out / "usage_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for k, val in summary.items():
        print(f"  {k:<32} {val}")
    print(f"written usage_per_scan.json and usage_summary.json to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("usage")
    u.add_argument("answers")
    u.add_argument("--out", required=True)
    u.add_argument("--train-check", action="store_true")
    args = ap.parse_args()
    return {"usage": cmd_usage}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
