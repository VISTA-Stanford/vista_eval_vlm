"""The library: a small wiki about telling five causes of lung change apart on chest CT (radiation, immunotherapy,
other drug, organizing pneumonia, infection), built from the published literature and from labelled scans of other
patients at our hospital. Spec planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-library.md.

    python3 -B tools/library.py search WORDS --scan CALL_DIR [--cause C] [--kind page|lit|example] [--limit 20]
    python3 -B tools/library.py page NAME --scan CALL_DIR          print a wiki page (start with: page index)
    python3 -B tools/library.py lit ID --scan CALL_DIR             a literature image: caption, source and its file
    python3 -B tools/library.py example ID --scan CALL_DIR [--out /tmp/lib]

CALL_DIR is your scan's folder: the one that holds ct/, organs.json and library_assets/.
    python3 -B tools/library.py check                      (for the builders) validate the library

What is where, inside the program folder and beside the reader's scan:
- library/pages/<name>.md: the wiki. Pages link to each other as [[name]], to literature images as [[lit:ID]] and to
  Stanford examples as [[ex:ID]]. library/catalogue/: what the builders found in our own data (counts of each pattern
  by cause).
- library_assets/lit/ beside the scan's ct/ folder: the literature images, and library_assets/lit/index.json (id,
  file, cause, pattern, caption, source, licence).
- library/examples/<id>/entry.json: a Stanford example (another patient's scan): cause, patterns, title, findings
  (text, slices, place), report_sentence, source, images [{image, view, marks}]. `example` draws each image beside the
  same region of YOUR scan, and prints where that example's full CT is (library_scans/<source>/ct beside your ct/) so
  you can view more of its slices with view_ct.py or zoom_organ.py.

Every command is noted in library_log.jsonl beside the scan's folder (what was opened, never how it looked).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import examples as ex  # noqa: E402  (the view format and the side-by-side drawing)

PROGRAM = Path(__file__).resolve().parents[1]
LIBRARY = PROGRAM / "library"
PAGES = LIBRARY / "pages"
EXAMPLES = LIBRARY / "examples"
CATALOGUE = LIBRARY / "catalogue"
CAUSES = ex.VARIANTS
MAX_EXAMPLES = 150
MAX_IMAGES_PER_EXAMPLE = 4
MAX_IMAGE_BYTES = 450_000
MAX_LIBRARY_TEXT = 1_500_000  # raised from 400 KB (Phil 2026-11-25: managers added ~100 KB each)
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
LINK = re.compile(r"\[\[(?:(lit|ex):)?([A-Za-z0-9][A-Za-z0-9._-]{0,99})\]\]")
SLICES = re.compile(r"^\d+(-\d+)?$")
LOG = "library_log.jsonl"
_call: Path | None = None


def log(entry: dict) -> None:
    if _call is None:
        return
    try:
        with (_call / LOG).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def find_call() -> Path | None:
    """The reader's call folder: the one that holds ct/ and library_assets/ (the cwd or its parents, or /workspace)."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "ct").is_dir() and (d / "library_assets").is_dir():
            return d
    return None


def lit_dir(call: Path | None) -> Path | None:
    return call / "library_assets" / "lit" if call else None


def lit_index(call: Path | None) -> list[dict]:
    d = lit_dir(call)
    f = d / "index.json" if d else None
    return json.loads(f.read_text(encoding="utf-8")) if f and f.is_file() else []


def pages() -> dict[str, str]:
    return {f.stem: f.read_text(encoding="utf-8") for f in sorted(PAGES.glob("*.md"))} if PAGES.is_dir() else {}


def hidden() -> set[str]:
    """Examples from this scan's own patient: hidden from this session (library_assets/exclude.json)."""
    f = _call / "library_assets" / "exclude.json" if _call else None
    return set(json.loads(f.read_text(encoding="utf-8"))) if f and f.is_file() else set()


def examples() -> dict[str, dict]:
    out, skip = {}, hidden()
    for f in sorted(EXAMPLES.glob("*/entry.json")) if EXAMPLES.is_dir() else []:
        if f.parent.name in skip:
            continue
        try:
            out[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out[f.parent.name] = {}
    return out


def example_text(e: dict) -> str:
    parts = [str(e.get(k, "")) for k in ("title", "report_sentence", "cause")]
    parts += [str(p) for p in e.get("patterns", [])]
    parts += [f"{x.get('text', '')} {x.get('place', '')}" for x in e.get("findings", []) if isinstance(x, dict)]
    return " ".join(parts)


def score(text: str, words: list[str]) -> int:
    t = text.lower()
    return sum(t.count(w) for w in words) if all(w in t for w in words) else 0


# --------------------------------------------------------------------------------------------------
# Reader commands
# --------------------------------------------------------------------------------------------------


def cmd_search(args) -> int:
    words = [w for w in args.words.lower().split() if w]
    hits = []
    if args.kind in (None, "page"):
        for name, text in pages().items():
            if args.cause and args.cause not in text.lower() and args.cause not in name:
                continue
            s = score(text + " " + name, words)
            if s:
                first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), name)
                hits.append((s, f"page {name}: {first}"))
    if args.kind in (None, "lit"):
        for r in lit_index(_call):
            if args.cause and r.get("cause") != args.cause:
                continue
            s = score(f"{r.get('pattern', '')} {r.get('caption', '')} {r.get('cause', '')}", words)
            if s:
                hits.append((s, f"lit {r['id']}: {r.get('cause')} / {r.get('pattern')}: {str(r.get('caption', ''))[:90]}"))
    if args.kind in (None, "example"):
        for eid, e in examples().items():
            if args.cause and e.get("cause") != args.cause:
                continue
            s = score(example_text(e), words)
            if s:
                hits.append((s, f"example {eid}: {e.get('cause')}: {e.get('title', '')}"))
    hits.sort(key=lambda h: (-h[0], h[1]))
    log({"cmd": "search", "words": args.words, "cause": args.cause, "kind": args.kind, "hits": len(hits)})
    if not hits:
        print("no matches" if (pages() or examples()) else "the library is empty")
        return 0
    print(f"{len(hits)} matches; showing {min(len(hits), args.limit)}")
    for _s, line in hits[: args.limit]:
        print(line)
    return 0


def cmd_page(args) -> int:
    text = pages().get(args.name)
    log({"cmd": "page", "name": args.name, "found": text is not None})
    if text is None:
        print(f"no page {args.name!r}; pages: {', '.join(sorted(pages())) or 'none'}")
        return 1
    print(text)
    return 0


def cmd_lit(args) -> int:
    r = next((x for x in lit_index(_call) if x.get("id") == args.id), None)
    log({"cmd": "lit", "id": args.id, "found": r is not None})
    if r is None:
        print(f"no literature image {args.id!r}; find one with: search WORDS --kind lit")
        return 1
    src = r.get("source", {})
    print(f"literature image {r['id']}: cause {r.get('cause')}, pattern {r.get('pattern')}")
    if r.get("panel"):
        print(f"panel(s) showing it: {r['panel']}")
    print(f"caption: {r.get('caption')}")
    print(f"source: {src.get('title', '')} ({src.get('journal', src.get('site or journal', ''))}, {src.get('year', '')})")
    print(f"image file: {lit_dir(_call) / r['file']}  (open it with Read)")
    return 0


def cmd_example(args) -> int:
    e = examples().get(args.id)
    log({"cmd": "example", "id": args.id, "found": e is not None})
    if e is None:
        print(f"no example {args.id!r}; find one with: search WORDS --kind example")
        return 1
    print(f"# {e.get('title')}\ncause (the answer key on that patient's scan): {e.get('cause')}")
    if e.get("patterns"):
        print(f"patterns: {', '.join(map(str, e['patterns']))}")
    for x in e.get("findings", []):
        print(f"- slices {x.get('slices')}{', ' + x['place'] if x.get('place') else ''}: {x.get('text', '')}")
    if e.get("report_sentence"):
        print(f"radiologist's words: {e['report_sentence']}")
    scan = Path(args.scan)
    organs_path = scan / "organs.json"
    organs = json.loads(organs_path.read_text(encoding="utf-8")) if organs_path.is_file() else None
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(e.get("images", []), 1):
        example = Image.open(EXAMPLES / args.id / im["image"]).convert("RGB")
        mine, why = ex.render(scan / "ct", organs, im.get("view", {}))
        out = args.out / f"{args.id}-{k}.jpg"
        if mine is None:
            example.save(out, format="JPEG", quality=85)
            print(f"image {k}: wrote {out} (the example only: {why})")
            continue
        ex.side_by_side(example.convert("L"), mine, f"EXAMPLE ({e.get('cause')})", "YOUR SCAN, same region").convert("RGB").save(
            out, format="JPEG", quality=85)
        print(f"image {k}: wrote {out} (left: the example, marked; right: your scan, same region)")
    full = scan / "library_scans" / str(e.get("source", "")) / "ct"
    print(f"the example's full CT: {full}" + ("" if full.is_dir() else " (not available for this scan)"))
    print("Open each image with Read; view more of the example's slices with view_ct.py on its full CT.")
    return 0


# --------------------------------------------------------------------------------------------------
# The builders' check
# --------------------------------------------------------------------------------------------------


def example_problems(eid: str, e: dict) -> list[str]:
    bad = []
    if not NAME.match(eid):
        bad.append("folder name: lowercase letters, digits and dashes")
    if not isinstance(e, dict) or not e:
        return bad + ["entry.json is missing or not a JSON object"]
    if e.get("cause") not in CAUSES:
        bad.append(f"cause must be one of {CAUSES}")
    for k in ("title", "source"):
        if not str(e.get(k, "")).strip():
            bad.append(f"{k} is required")
    f = e.get("findings")
    if not isinstance(f, list) or not 1 <= len(f) <= 8 or not all(
            isinstance(x, dict) and str(x.get("text", "")).strip() and SLICES.match(str(x.get("slices", ""))) for x in f):
        bad.append("findings: 1 to 8 of {text, slices (\"38\" or \"38-52\": the source scan's axial slices), place}")
    ims = e.get("images")
    if not isinstance(ims, list) or not 1 <= len(ims) <= MAX_IMAGES_PER_EXAMPLE:
        bad.append(f"images: 1 to {MAX_IMAGES_PER_EXAMPLE} of {{image, view, marks}}")
        ims = []
    for im in ims:
        p = EXAMPLES / eid / str(im.get("image", "")) if isinstance(im, dict) else None
        if p is None or not im.get("image") or "/" in str(im["image"]) or not p.is_file():
            bad.append("each image names a file in the example's folder")
            continue
        if p.stat().st_size > MAX_IMAGE_BYTES:
            bad.append(f"{im['image']} is over {MAX_IMAGE_BYTES // 1000} KB: draw it with atlas.py draw (768 px JPEG)")
        v = im.get("view")
        q = ex.view_problem(v) if v is not None else "each image records its view (how it was drawn)"
        if q:
            bad.append(q)
    return bad


def library_problems(lit_ids: set[str] | None = None) -> list[str]:
    """Problems with the library: pages, links, examples, size. ``lit_ids``: the literature image ids that exist."""
    ps, exs = pages(), examples()
    bad = []
    if "index" not in ps:
        bad.append("library/pages/index.md (the start-here page) is missing")
    for name in ps:
        if not NAME.match(name):
            bad.append(f"page {name}: names are lowercase letters, digits and dashes")
    for name, text in ps.items():
        for kind, target in LINK.findall(text):
            if kind == "" and target not in ps:
                bad.append(f"page {name}: link [[{target}]] has no page")
            elif kind == "ex" and target not in exs:
                bad.append(f"page {name}: link [[ex:{target}]] has no example")
            elif kind == "lit" and lit_ids is not None and target not in lit_ids:
                bad.append(f"page {name}: link [[lit:{target}]] has no literature image")
    if len(exs) > MAX_EXAMPLES:
        bad.append(f"{len(exs)} examples; at most {MAX_EXAMPLES}")
    for eid, e in exs.items():
        bad += [f"example {eid}: {p}" for p in example_problems(eid, e)]
    size = sum(f.stat().st_size for f in LIBRARY.rglob("*") if f.is_file() and f.suffix in (".md", ".json"))
    if size > MAX_LIBRARY_TEXT:
        bad.append(f"the library's text is {size // 1000} KB; keep it under {MAX_LIBRARY_TEXT // 1000} KB")
    return bad


def cmd_check(args) -> int:
    ids = None
    if args.lit_index:
        ids = {r["id"] for r in json.loads(Path(args.lit_index).read_text(encoding="utf-8"))}
    bad = library_problems(ids)
    for p in bad[:40]:
        print(p)
    if bad:
        print(f"FAIL: {len(bad)} problems")
        return 1
    print(f"PASS: {len(pages())} pages, {len(examples())} examples")
    return 0


def main(argv=None) -> int:
    global _call
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("words")
    s.add_argument("--scan", type=Path)
    s.add_argument("--cause", choices=CAUSES)
    s.add_argument("--kind", choices=("page", "lit", "example"))
    s.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("page")
    p.add_argument("name")
    p.add_argument("--scan", type=Path)
    li = sub.add_parser("lit")
    li.add_argument("id")
    li.add_argument("--scan", type=Path)
    e = sub.add_parser("example")
    e.add_argument("id")
    e.add_argument("--scan", type=Path, required=True)
    e.add_argument("--out", type=Path, default=Path("/tmp/lib"))
    c = sub.add_parser("check")
    c.add_argument("--lit-index", help="the literature images' index.json, to check [[lit:ID]] links")
    args = ap.parse_args(argv)
    _call = Path(args.scan).resolve() if getattr(args, "scan", None) else find_call()
    return {"search": cmd_search, "page": cmd_page, "lit": cmd_lit, "example": cmd_example, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
