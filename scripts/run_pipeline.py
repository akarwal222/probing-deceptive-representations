"""
End-to-end entrypoint: extract activations, then train + analyze probes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    steps = [
        [sys.executable, "-m", "scripts.extract_activations", "--config", args.config],
        [sys.executable, "-m", "scripts.train_probes", "--config", args.config],
    ]
    for step in steps:
        print(f"\n{'=' * 60}\nRunning: {' '.join(step)}\n{'=' * 60}")
        result = subprocess.run(step)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
