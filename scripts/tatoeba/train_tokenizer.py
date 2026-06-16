from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tokenizer import main


DEFAULT_CONFIG = "configs/tatoeba_config.yaml"


def _ensure_config_arg(argv: list[str]) -> list[str]:
    if "--config" in argv or any(arg.startswith("--config=") for arg in argv):
        return argv
    return [argv[0], "--config", DEFAULT_CONFIG, *argv[1:]]


if __name__ == "__main__":
    sys.argv = _ensure_config_arg(sys.argv)
    raise SystemExit(main())
