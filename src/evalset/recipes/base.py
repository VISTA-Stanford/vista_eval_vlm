"""What a model recipe is: how one model is shown a scan and asked a card's question.

A recipe fixes the render (slice sampler and count, windows, size, file format, orientation) and the
wrapper (slice labels, where the question goes, how the answer is forced). The render settings name
the image pack (``render_key``), so two recipes that render alike share one pack on the mount.

Messages are built as a list of parts, images first and the question last, so every question about
one scan shares the same prefix and the model's cache can reuse it (``Message.prefix_len``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from context.windows import describe

YES_NO = ("yes", "no")


@dataclasses.dataclass(frozen=True)
class Message:
    """One call: ``parts`` in order; the first ``prefix_len`` parts are the cacheable images + labels."""
    parts: tuple
    prefix_len: int

    @property
    def images(self):
        return [p["data"] for p in self.parts if p["type"] == "image"]

    @property
    def question(self):
        return self.parts[-1]["text"]


@dataclasses.dataclass(frozen=True)
class Recipe:
    name: str
    model_type: str           # models.MODEL_REGISTRY key, or "vllm" for the open-model runner
    model_name: str
    n_slices: int             # per scan, one-scan cards (slice count is the pilot's to set)
    n_slices_pair: int        # per scan, two-scan cards (prior + current)
    sampler: str = "medgemma_uniform"
    windows: str = "medgemma"
    size: int | None = None   # None = native (512 for the feb26 store); else square resize
    image_format: str = "jpeg"
    jpeg_quality: int | None = None  # None = PIL's default (75), as the MedGemma notebook encodes
    orientation: str = "radiological"
    label_style: str = "before"      # "before": "Image n:" then the image; "after": image then "SLICE n"
    answer_form: str = "schema"      # "schema": forced {"answer": yes|no}; "final_answer": free text
    persona: str = ""                # line put before the card text (CT arms only)
    answer_line: str = ""            # replaces the card's "Answer yes or no." when set
    max_images: int | None = None
    settings: tuple = ()             # model call settings, as (key, value) pairs: effort, decoding...

    # -- render ------------------------------------------------------------------------------
    def render_key(self, n: int) -> dict:
        """The settings that decide the pixels, for ``n`` slices per scan."""
        return {"sampler": self.sampler, "k": n, "windows": self.windows, "size": self.size,
                "format": self.image_format, "quality": self.jpeg_quality, "orientation": self.orientation}

    def pack_id(self, n: int) -> str:
        key = self.render_key(n)
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
        size = "native" if self.size is None else f"{self.size}px"
        q = "" if self.image_format != "jpeg" else f"q{self.jpeg_quality or 75}"
        return f"{self.sampler}{n}_{self.windows}_{size}_{self.image_format}{q}_{self.orientation}_{h}"

    def slices_per_scan(self, n_scans: int) -> int:
        return self.n_slices if n_scans == 1 else self.n_slices_pair

    # -- wrapper -----------------------------------------------------------------------------
    def windows_phrase(self) -> str:
        return describe(self.windows)

    def _label(self, scan_name: str | None, j: int) -> str:
        if self.label_style == "after":
            return f"{scan_name} SLICE {j}" if scan_name else f"SLICE {j}"
        return f"{scan_name} scan, image {j}:" if scan_name else f"Image {j}:"

    def wrap_text(self, card_text: str, with_images: bool) -> str:
        text = card_text
        if self.answer_line:
            if not text.endswith("Answer yes or no."):
                raise ValueError("card text must end with 'Answer yes or no.'")
            text = text[: -len("Answer yes or no.")] + self.answer_line
        if with_images and self.persona:
            text = self.persona + "\n" + text
        return text

    def build(self, card_text: str, scans: list[tuple[str | None, list[bytes]]] | None) -> Message:
        """One call. ``scans``: [(scan name or None, [image bytes, ...]), ...] in image order, or None
        for a text-only arm. Raises when the images exceed the model's cap."""
        mime = "image/jpeg" if self.image_format == "jpeg" else "image/png"
        parts = []
        for scan_name, images in scans or []:
            for j, data in enumerate(images, 1):
                label = {"type": "text", "text": self._label(scan_name, j)}
                img = {"type": "image", "data": data, "mime": mime}
                parts += [img, label] if self.label_style == "after" else [label, img]
        n_img = sum(1 for p in parts if p["type"] == "image")
        if self.max_images is not None and n_img > self.max_images:
            raise ValueError(f"{self.name}: {n_img} images, cap is {self.max_images}")
        prefix = len(parts)
        parts.append({"type": "text", "text": self.wrap_text(card_text, bool(scans))})
        return Message(parts=tuple(parts), prefix_len=prefix)

    def setting(self, key, default=None):
        return dict(self.settings).get(key, default)

    def replace(self, **kw) -> "Recipe":
        return dataclasses.replace(self, **kw)
