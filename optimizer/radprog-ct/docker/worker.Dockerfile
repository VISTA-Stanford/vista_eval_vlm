# The sealed worker's image: the labeling program's runtime and nothing else. No task code, no data:
# the program and worker.py are mounted read-only at run time, so changing them needs no rebuild.
# Pinned by the multi-platform index digest (read from Docker Hub on 2026-10-16) so a rebuild can't pick
# up a newer base silently. If your program needs packages, add them here (pin their versions too).
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
# RUN pip install --no-cache-dir 'spacy==3.7.5'
