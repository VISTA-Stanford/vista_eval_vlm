"""Pre-run spot check for the eval set: build exactly what each model would receive for one real case per step
and check it, without calling any model.

For every recipe x step x arm (one check-run case each, from the image packs already on the mount):
  * images per call match the recipe (fewer only when the scan has fewer slices), labels in the recipe's
    style and order (prior scan first), every image a decodable RGB image in the recipe's format (JPEG or PNG)
    at the expected size;
  * slices run head to feet (the stored index falls), the spine sits in the lower half of a mid slice, the first
    sent image re-renders byte for byte from the NIfTI, and it is much closer to that render than to its mirror;
    the JPEG error per window channel is reported (information, not a check);
  * the question is the card text wrapped by the recipe (persona / answer line), the image line names the
    recipe's windows, and no identifier, label value or report text appears in any prompt;
  * the request body each provider would get (Vertex Gemini, AI Hub, vLLM chat) has the expected shape.
It also reads measured tokens per call from finished runs (default run ``check5``).

    cd src && python ../scripts/evalset_spotcheck.py --out <json>

Prints one line per check (PASS / FAIL) and exits 1 on any FAIL. Writes aggregate results only: no ids.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context import nifti, windows  # noqa: E402
from evalset import packs, paths, run, samples  # noqa: E402
from evalset.cards import CARDS  # noqa: E402
from evalset.recipes import RECIPES  # noqa: E402

RESULTS = []
JPEG_ERR = {name: [] for name in ("medgemma", "astra", "argon")}


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))


def leaks(text: str, row: dict) -> list[str]:
    """Identifier-like or answer-key values from the case row that appear in the prompt text."""
    bad = []
    for k in ("case_id", "person_id", "ct_accession_number", "study", "series", "prior_study", "prior_series"):
        v = row.get(k)
        if v is not None and str(v) not in ("", "nan", "None") and len(str(v)) >= 5 and str(v) in text:
            bad.append(k)
    return bad


def spine_row_fraction(hu: np.ndarray) -> float:
    r, _ = np.nonzero(hu > 300)
    return float(r.mean() / hu.shape[0]) if len(r) else float("nan")


def image_checks(recipe, card, row, step):
    n = recipe.slices_per_scan(len(card.scan_names))
    scans = [("current", row["study"], row["series"])]
    if len(card.scan_names) == 2:
        scans = [("prior", row["prior_study"], row["prior_series"])] + scans
    for role, study, series in scans:
        m = json.loads((packs.pack_dir(recipe, n) / packs.scan_id(study, series) / "manifest.json").read_text())
        want = min(n, m["n_slices_in_volume"])
        tag = f"{recipe.name} step {step} {role} scan"
        check(f"{tag}: {len(m['files'])} slices sent (recipe {n}, volume {m['n_slices_in_volume']})", len(m["files"]) == want)
        v = m["voxel_indices"]
        check(f"{tag}: head to feet (stored index falls)", all(a > b for a, b in zip(v, v[1:])))
        imgs = packs.load(recipe, n, study, series)
        pil = [Image.open(io.BytesIO(b)) for b in imgs]
        sizes = {p.size for p in pil}
        fmts = {(p.format, p.mode) for p in pil}
        check(f"{tag}: images {sorted(fmts)} at {sorted(sizes)}",
              fmts == {(recipe.image_format.upper(), "RGB")} and len(sizes) == 1)
        stack = nifti.AxialStack.open(nifti.ct_path(study, series, mount_root=paths.mount_root()))
        k0 = m["display_indices"][0]
        fresh = windows.rgb_u8(stack[k0], recipe.windows).astype(float)
        sent = np.asarray(pil[0].convert("RGB"), dtype=float)
        same = packs.encode(windows.rgb_u8(stack[k0], recipe.windows), recipe) == imgs[0]
        check(f"{tag}: first image re-renders byte for byte from the NIfTI", same)
        if fresh.shape == sent.shape:
            per_ch = np.abs(fresh - sent).mean(axis=(0, 1))
            err, err_mirror = float(per_ch.mean()), float(np.abs(fresh[:, ::-1] - sent).mean())
            check(f"{tag}: orientation — sent image is closer to the render than to its mirror ({err:.1f} vs {err_mirror:.1f})",
                  err < err_mirror / 2)
            JPEG_ERR[recipe.name].append(per_ch.round(1).tolist())
        mid = stack[m["display_indices"][len(m["display_indices"]) // 2]]
        f = spine_row_fraction(mid)
        check(f"{tag}: spine in lower half of mid slice (row fraction {f:.2f})", f > 0.5)


def wire_checks(recipe, msg, step, arm):
    tag = f"{recipe.name} step {step} {arm}"
    parts = list(msg.parts)
    if recipe.model_type == "vertex_gemini":
        from models.vertex_gemini import build_parts_request
        b = build_parts_request(parts, run.CHOICES, int(recipe.setting("max_new_tokens", 8192)),
                                recipe.setting("media_resolution"), recipe.setting("thinking_level"))
        kinds = [next(iter(p)) for p in b["contents"][0]["parts"]]
        g = b["generationConfig"]
        ok = kinds[-1] == "text" and g["responseSchema"]["enum"] == ["yes", "no"] and \
            g.get("mediaResolution") == recipe.setting("media_resolution") and \
            g.get("thinkingConfig", {}).get("thinkingLevel") == recipe.setting("thinking_level")
        check(f"{tag}: Vertex body — {kinds.count('inlineData')} images, question last, yes/no enum, "
              f"{g.get('mediaResolution')}, thinking {g.get('thinkingConfig', {}).get('thinkingLevel')}", ok)
        return {"provider": "Vertex generateContent", "parts": f"{kinds.count('text') - 1} labels + {kinds.count('inlineData')} images, then question",
                "generationConfig": {k: v for k, v in g.items() if k != "responseSchema"} | {"responseSchema": "enum yes/no"}}
    if recipe.model_type == "aihub_openai":
        from models.aihub_openai import build_parts_request
        b = build_parts_request(parts, run.CHOICES, int(recipe.setting("max_new_tokens", 4096)),
                                recipe.setting("reasoning_effort"))
        c = b["messages"][0]["content"]
        n_img = sum(x["type"] == "image_url" for x in c)
        props = b["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"]
        ok = c[-1]["type"] == "text" and props == ["yes", "no"] and n_img <= 50 and \
            b.get("reasoning_effort") == recipe.setting("reasoning_effort")
        check(f"{tag}: AI Hub body — {n_img} images (cap 50), question last, strict schema yes/no, effort {b.get('reasoning_effort')}", ok)
        return {"provider": "AI Hub chat/completions", "parts": f"{n_img} labels + {n_img} images, then question",
                "settings": {"reasoning_effort": b.get("reasoning_effort"), "max_completion_tokens": b["max_completion_tokens"],
                             "response_format": "strict JSON schema {answer: yes|no}"}}
    c = run.openai_content(msg)
    kinds = [x["type"] for x in c]
    ok = kinds[-1] == "text" and all(kinds[i] == "image_url" and kinds[i + 1] == "text" for i in range(0, len(kinds) - 1, 2))
    check(f"{tag}: vLLM chat — {kinds.count('image_url')} images each followed by its label, question last", ok)
    return {"provider": "vLLM chat (weill GPU)", "parts": f"{kinds.count('image_url')} x (image, label), then question",
            "settings": {k: v for k, v in recipe.settings}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="check5")
    args = ap.parse_args()
    examples, tokens = {}, {}
    for rname, recipe in RECIPES.items():
        for step, card in CARDS.items():
            row = packs.check_cases(samples.load(step), step, 1).iloc[0].to_dict()
            image_checks(recipe, card, row, step)
            for arm in card.arms:
                qm = run.messages_for_case(recipe, card, row, arm)
                q, msg = qm[0]
                want_text = recipe.wrap_text(card.text(q, recipe.windows_phrase(), arm, summary=row.get("summary")), arm == "ct")
                tag = f"{rname} step {step} {arm}"
                check(f"{tag}: question = card text wrapped by the recipe", msg.question == want_text)
                if arm == "ct":
                    check(f"{tag}: image line names {recipe.windows_phrase()}", recipe.windows_phrase() in msg.question)
                    labels = [p["text"] for p in msg.parts[:-1] if p["type"] == "text"]
                    names = [s for s in card.scan_names if s]
                    order_ok = (not names) or labels[0].startswith(names[0])
                    check(f"{tag}: labels '{labels[0]}' … '{labels[-1]}' ({len(labels)}), prior first", order_ok and len(labels) == len(msg.images))
                else:
                    check(f"{tag}: no images; summary text included", not msg.images and str(row.get("summary", ""))[:40] in msg.question)
                lk = sorted({k for _, m in qm for k in leaks(m.question, row)})
                check(f"{tag}: no identifiers in any of its {len(qm)} prompts", not lk, ", ".join(lk))
                if arm == "ct":
                    for key in ("ct_note_text", "report"):
                        v = row.get(key)
                        if isinstance(v, str) and len(v) > 40:
                            check(f"{tag}: report text not in prompt", v[:40] not in msg.question)
                wire = wire_checks(recipe, msg, step, arm)
                examples[f"{rname}|{step}|{arm}"] = {
                    "n_images": len(msg.images), "labels": [p["text"] for p in msg.parts[:-1] if p["type"] == "text"][:2]
                    + (["…"] if len(msg.parts) > 6 else []) + [p["text"] for p in msg.parts[:-1] if p["type"] == "text"][-1:],
                    "question": msg.question, "wire": wire, "n_questions": len(qm)}
        for f in sorted(glob.glob(str(paths.root() / "runs" / args.run / rname / "step*_ct.csv"))):
            import pandas as pd
            d = pd.read_csv(f)
            tokens[f"{rname}|{Path(f).stem}"] = {"calls": len(d), "prompt_tokens_per_call": round(float(d["usage_prompt_tokens"].mean())),
                                                  "parsed": int(d["answer"].isin(["yes", "no"]).sum()),
                                                  "yes_share": round(float((d["answer"] == "yes").mean()), 2)}
    n_fail = sum(not r["ok"] for r in RESULTS)
    Path(args.out).write_text(json.dumps({"checks": RESULTS, "examples": examples, "tokens": tokens, "jpeg_error_per_channel": JPEG_ERR,
                                          "recipes": {k: {f: (dict(v) if f == "settings" else v) for f, v in vars(r).items()}
                                                      for k, r in RECIPES.items()}}, indent=1, default=str))
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed -> {args.out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
