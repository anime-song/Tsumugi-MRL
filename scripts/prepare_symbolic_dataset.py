"""Compatibility entry point; implementation lives in train.symbolic_teacher."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.symbolic_teacher.prepare_dataset import main

if __name__ == "__main__":
    main()
