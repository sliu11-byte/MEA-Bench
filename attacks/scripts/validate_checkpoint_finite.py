from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.core.checkpoint_health import checkpoint_tensor_health


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail unless every tensor in a model/adapter checkpoint is finite.")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    health = checkpoint_tensor_health(args.checkpoint)
    print(json.dumps(health, ensure_ascii=False, indent=2))
    return 0 if health["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
