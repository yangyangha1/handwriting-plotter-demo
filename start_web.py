"""Run this before installing dependencies: environment UI uses only stdlib."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from hwplotter.webui import main

if __name__ == "__main__":
    main()
