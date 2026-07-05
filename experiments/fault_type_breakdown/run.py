from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hra_sl.fault_types import main


if __name__ == "__main__":
    main()
