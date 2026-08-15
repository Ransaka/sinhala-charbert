"""
CLI script to synthesize noisy training data from clean datasets (e.g., Ransaka/sinhala-450M-sample).
"""

import argparse
import json
from pathlib import Path
from tabulate import tabulate
from sinhala_charbert.config.noise_config import NoiseProfile
from sinhala_charbert.data.dataset_builder import build_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate SynTypo-SI synthetic typo dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Ransaka/sinhala-450M-sample",
        help="HuggingFace dataset repository or local path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to process.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/syntypo_sample_pairs.jsonl",
        help="Output JSONL path for synthetic pairs.",
    )
    args = parser.parse_args()

    print(f"Loading dataset '{args.dataset_name}' (split: {args.split}, samples: {args.num_samples})...")
    profile = NoiseProfile()
    ds = build_synthetic_dataset(
        dataset_name_or_path=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        noise_profile=profile,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(ds)} samples to {out_path}...")
    records = []
    table_rows = []
    for idx, sample in enumerate(ds):
        records.append({
            "id": f"syn_si_{idx:07d}",
            "source_noisy": sample["source_noisy"],
            "target_clean": sample["target_clean"],
            "error_ops": sample["error_ops"],
            "has_error": sample["has_error"],
        })
        if idx < 5:
            table_rows.append([
                idx + 1,
                sample["target_clean"][:40] + ("..." if len(sample["target_clean"]) > 40 else ""),
                sample["source_noisy"][:40] + ("..." if len(sample["source_noisy"]) > 40 else ""),
                ", ".join(sample["error_ops"][:2]) if sample["error_ops"] else "CLEAN",
            ])

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nSample Synthesized Pairs:")
    print(tabulate(table_rows, headers=["#", "Target Clean", "Synthesized Noisy", "Sample Operations"], tablefmt="grid"))
    print(f"\nSuccessfully generated {len(records)} pairs saved to {out_path}")


if __name__ == "__main__":
    main()
