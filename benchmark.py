#!/usr/bin/env python3
from pathlib import Path
import sys

# Allow running directly from a source checkout before editable installation.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference_bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
