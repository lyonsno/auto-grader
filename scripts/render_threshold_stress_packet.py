from __future__ import annotations

import argparse
import json

from auto_grader.paper_calibration_packet import write_mc_threshold_stress_packet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a printable two-page MC/OpenCV threshold-stress packet with direct per-question marking instructions."
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/mc-threshold-stress-packet",
        help="Directory where the packet PDF and JSON artifacts should be written.",
    )
    parser.add_argument(
        "--seed",
        default="17",
        help="Deterministic seed for packet generation.",
    )
    args = parser.parse_args()

    result = write_mc_threshold_stress_packet(output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
