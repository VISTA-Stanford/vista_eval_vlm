import sys
from pathlib import Path

# The repo's modules import as top-level packages from src/ (e.g. `from models import ...`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
