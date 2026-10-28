"""The frontier-model eval set: one pipeline from a task card + a model recipe to calls and scores.

Spec: planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md (on the shared mount).
``recipes/`` how each model is shown a scan; ``cards/`` the questions; ``packs`` the rendered slices
on the mount; ``samples`` the fixed patient/scan lists; ``run`` the calls; ``score`` the tables.
"""
