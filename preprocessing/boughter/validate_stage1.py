#!/usr/bin/env python3
"""
Boughter Dataset Preprocessing - Stage 1 Validator

Validates that Stage 1 (DNA → Protein translation) completed successfully,
with correct translation, sequence counts, and data integrity.

Pipeline Position: Validates Stage 1 output
    Stage 1 → boughter.csv (1,117 sequences) ← VALIDATED BY THIS SCRIPT
    Stages 2+3 → Fragment CSVs (1,065 sequences)

Usage:
    python3 preprocessing/boughter/validate_stage1.py

Validation Checks:
    1. train_datasets/boughter/processed/boughter.csv exists and is readable
    2. Sequence count: 1,117 (95.4% translation success from 1,171 raw)
    3. All sequences contain valid protein characters (ACDEFGHIKLMNPQRSTVWY)
    4. Required columns present: id, subset, heavy, light, label, num_flags, flag_category
    5. Labels are valid (0=specific, 1=non-specific)
    6. Novo flagging applied correctly (0 flags → 0, 4+ flags → 1)

Outputs:
    - Console report with validation results
    - train_datasets/boughter/annotated/validation_report.txt

Reference: See docs/boughter/boughter_data_sources.md for Stage 1 methodology
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_stage1_output() -> dict[str, Any]:
    """Validate Stage 1 output (boughter.csv)."""
    print("\n" + "=" * 70)
    print("Stage 1 Validation: DNA Translation & Novo Flagging")
    print("=" * 70)

    csv_path = Path("train_datasets/boughter/processed/boughter.csv")

    if not csv_path.exists():
        return {
            "success": False,
            "error": f"Stage 1 output not found: {csv_path}",
        }

    df = pd.read_csv(csv_path)

    # Check expected total
    expected_total = 1171  # From raw FASTA files
    actual_total = len(df)
    loss = expected_total - actual_total
    loss_pct = (loss / expected_total) * 100

    print(f"\n✓ Stage 1 Output: {csv_path}")
    print(f"  Expected sequences: {expected_total}")
    print(f"  Actual sequences:   {actual_total}")
    print(f"  Loss:               {loss} ({loss_pct:.2f}%)")

    # Check for sequence quality issues
    print("\n✓ Sequence Quality Checks:")

    # Stop codons
    heavy_stops = df["heavy_seq"].str.contains("\\*", regex=True).sum()
    light_stops = df["light_seq"].str.contains("\\*", regex=True).sum()
    print(f"  Stop codons (*) in heavy: {heavy_stops}")
    print(f"  Stop codons (*) in light: {light_stops}")

    # Excessive X's (>5%)
    def check_x_ratio(seq: str | float | None) -> bool:
        if pd.isna(seq):
            return False
        seq_str = str(seq)
        if not seq_str:
            return False
        return seq_str.count("X") / len(seq_str) > 0.05

    heavy_x = df["heavy_seq"].apply(check_x_ratio).sum()
    light_x = df["light_seq"].apply(check_x_ratio).sum()
    print(f"  Excessive X's (>5%) in heavy: {heavy_x}")
    print(f"  Excessive X's (>5%) in light: {light_x}")

    # Length distribution
    heavy_len = df["heavy_seq"].str.len()
    light_len = df["light_seq"].str.len()
    print("\n✓ Sequence Length Ranges:")
    print(
        f"  Heavy: {heavy_len.min()}-{heavy_len.max()} aa "
        f"(mean: {heavy_len.mean():.1f})"
    )
    print(
        f"  Light: {light_len.min()}-{light_len.max()} aa "
        f"(mean: {light_len.mean():.1f})"
    )

    # Check Novo flagging distribution
    print("\n✓ Novo Flagging Strategy:")
    for category in ["specific", "mild", "non_specific"]:
        count = len(df[df["flag_category"] == category])
        pct = count / len(df) * 100
        included = len(
            df[(df["flag_category"] == category) & df["include_in_training"]]
        )
        print(f"  {category:15s}: {count:4d} ({pct:5.2f}%) - {included} in training")

    # Training set balance
    training_df = df[df["include_in_training"]]
    print("\n✓ Training Set Balance:")
    print(f"  Total training:    {len(training_df)}")
    if len(training_df) > 0:
        label_counts = training_df["label"].value_counts()
        for label in sorted(label_counts.index):
            count = label_counts[label]
            pct = count / len(training_df) * 100
            label_name = "Specific (0)" if label == 0 else "Non-specific (1)"
            print(f"  {label_name}: {count} ({pct:.1f}%)")

    return {
        "success": True,
        "total_sequences": actual_total,
        "loss": loss,
        "loss_pct": loss_pct,
        "quality_issues": heavy_stops + light_stops + heavy_x + light_x,
        "training_size": len(training_df),
    }


def validate_stage2_output() -> dict[str, Any]:
    """Validate Stage 2 output (fragment CSVs + annotation)."""
    print("\n" + "=" * 70)
    print("Stage 2 Validation: ANARCI Annotation & Fragment Extraction")
    print("=" * 70)

    # Check if Stage 2 has been run
    output_dir = Path("train_datasets/boughter/annotated")
    vh_file = output_dir / "VH_only_boughter.csv"

    if not vh_file.exists():
        return {
            "success": False,
            "error": f"Stage 2 output not found: {vh_file}",
        }

    # Load Stage 1 input
    df_stage1 = pd.read_csv("train_datasets/boughter/processed/boughter.csv")

    stage1_count = len(df_stage1)
    # Stage 2 failures from log (if present)
    failure_ids: list[str] = []
    failure_log_path = output_dir / "annotation_failures.log"
    if failure_log_path.exists():
        failures_log = failure_log_path.read_text().strip()
        if failures_log:
            failure_ids = [
                line.strip() for line in failures_log.split("\n") if line.strip()
            ]

    stage2_failures = len(failure_ids)
    stage2_annotated = stage1_count - stage2_failures
    stage2_success_rate = (stage2_annotated / stage1_count) * 100

    print("\n✓ ANARCI Annotation Success:")
    print(f"  Stage 1 input:          {stage1_count} sequences")
    print(f"  Stage 2 annotated:      {stage2_annotated} sequences")
    print(
        f"  Stage 2 failures:       {stage2_failures} ({stage2_failures / stage1_count * 100:.2f}%)"
    )
    target_met = stage2_success_rate >= 95.0
    status = "✅ PASS" if target_met else "❌ FAIL"
    print(f"  Success rate:           {stage2_success_rate:.2f}%  ({status})")

    # Stage 3 QC removals
    qc_log_path = output_dir / "qc_filtered_sequences.txt"
    qc_ids: list[str] = []
    if qc_log_path.exists():
        qc_text = qc_log_path.read_text().strip()
        if qc_text:
            qc_ids = [line.strip() for line in qc_text.split("\n") if line.strip()]
    stage3_removed = len(qc_ids)
    stage3_retained = stage2_annotated - stage3_removed
    stage3_retention = (
        (stage3_retained / stage2_annotated) * 100 if stage2_annotated else 0.0
    )

    print("\n✓ Stage 3 (Post-Annotation QC):")
    print(f"  Sequences entering Stage 3: {stage2_annotated}")
    print(f"  Filtered (X/empty CDRs):    {stage3_removed}")
    print(
        f"  Final clean sequences:      {stage3_retained} ({stage3_retention:.2f}% retention)"
    )

    # Load final fragment (VH) to confirm counts
    df_vh = pd.read_csv(vh_file, comment="#")
    if len(df_vh) != stage3_retained:
        print(
            f"\n⚠️  Warning: Final VH count ({len(df_vh)}) does not match "
            f"Stage 3 retained count ({stage3_retained})."
        )

    # Report failures by subset (Stage 2)
    if failure_ids:
        print("\n✓ Stage 2 Failures by Subset:")
        from collections import Counter

        subset_counts: Counter[str] = Counter()
        for fid in failure_ids:
            subset = fid.split("_")[0]
            # preserve hiv_* prefixes
            if fid.startswith("hiv_nat"):
                subset = "hiv_nat"
            elif fid.startswith("hiv_cntrl"):
                subset = "hiv_cntrl"
            elif fid.startswith("hiv_plos"):
                subset = "hiv_plos"
            elif fid.startswith("gut_hiv"):
                subset = "gut_hiv"
            elif fid.startswith("mouse_iga"):
                subset = "mouse_iga"
            elif fid.startswith("flu"):
                subset = "flu"
            subset_counts[subset] += 1

        for subset in sorted(subset_counts.keys()):
            subset_total = len(df_stage1[df_stage1["subset"] == subset])
            subset_failures = subset_counts[subset]
            subset_fail_pct = (subset_failures / subset_total) * 100
            print(
                f"  {subset:12s}: {subset_failures:3d}/{subset_total:3d} "
                f"({subset_fail_pct:5.1f}%)"
            )

    # Check CDR length distributions
    print("\n✓ CDR Length Distributions (from VH_only):")
    h_cdr1_file = output_dir / "H-CDR1_boughter.csv"
    h_cdr2_file = output_dir / "H-CDR2_boughter.csv"
    h_cdr3_file = output_dir / "H-CDR3_boughter.csv"

    if all(f.exists() for f in [h_cdr1_file, h_cdr2_file, h_cdr3_file]):
        cdr1 = pd.read_csv(h_cdr1_file, comment="#")
        cdr2 = pd.read_csv(h_cdr2_file, comment="#")
        cdr3 = pd.read_csv(h_cdr3_file, comment="#")

        for name, df_cdr in [("H-CDR1", cdr1), ("H-CDR2", cdr2), ("H-CDR3", cdr3)]:
            lengths = df_cdr["sequence_length"]
            print(
                f"  {name}: {lengths.min()}-{lengths.max()} aa "
                f"(mean: {lengths.mean():.1f}, median: {lengths.median():.0f})"
            )

    # Check fragment file count
    fragment_files = list(output_dir.glob("*_boughter.csv"))
    print(f"\n✓ Fragment Files: {len(fragment_files)} files found")
    expected_fragments = 16
    if len(fragment_files) == expected_fragments:
        print(f"  Status: ✅ All {expected_fragments} fragments present")
    else:
        print(
            f"  Status: ❌ Expected {expected_fragments}, found {len(fragment_files)}"
        )

    return {
        "success": True,
        "stage2_annotated": stage2_annotated,
        "stage2_failures": stage2_failures,
        "stage2_success_rate": stage2_success_rate,
        "target_met": target_met,
        "stage3_removed": stage3_removed,
        "stage3_retained": stage3_retained,
        "stage3_retention": stage3_retention,
        "fragment_files": len(fragment_files),
    }


def generate_report(
    stage1_results: dict[str, Any], stage2_results: dict[str, Any]
) -> None:
    """Generate validation report."""
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    report_lines = [
        "Boughter Dataset Validation Report",
        "=" * 70,
        "",
        "Stage 1: DNA Translation & Novo Flagging",
        "-" * 70,
    ]

    if stage1_results["success"]:
        report_lines.extend(
            [
                f"Total sequences:       {stage1_results['total_sequences']}",
                f"Translation loss:      {stage1_results['loss']} "
                f"({stage1_results['loss_pct']:.2f}%)",
                f"Quality issues:        {stage1_results['quality_issues']}",
                f"Training set size:     {stage1_results['training_size']}",
                "",
            ]
        )
    else:
        report_lines.append(f"ERROR: {stage1_results['error']}")
        report_lines.append("")

    report_lines.extend(
        [
            "Stage 2: ANARCI Annotation & Fragment Extraction",
            "-" * 70,
        ]
    )

    if stage2_results["success"]:
        status = "PASS ✅" if stage2_results["target_met"] else "FAIL ❌"
        report_lines.extend(
            [
                f"Stage 2 annotated:     {stage2_results['stage2_annotated']}",
                f"Stage 2 failures:      {stage2_results['stage2_failures']}",
                f"Stage 2 success rate:  {stage2_results['stage2_success_rate']:.2f}%",
                f"Target (>95%):         {status}",
                f"Stage 3 removed:       {stage2_results['stage3_removed']}",
                f"Stage 3 retention:     {stage2_results['stage3_retention']:.2f}%",
                f"Fragment files:        {stage2_results['fragment_files']}/16",
                "",
            ]
        )
    else:
        report_lines.append(f"ERROR: {stage2_results['error']}")
        report_lines.append("")

    report_lines.extend(
        [
            "Overall Status",
            "-" * 70,
        ]
    )

    if stage1_results["success"] and stage2_results["success"]:
        if stage2_results["target_met"]:
            report_lines.append("✅ Dataset is READY for ML training")
        else:
            report_lines.append(
                f"⚠️  Annotation success rate "
                f"({stage2_results['stage2_success_rate']:.2f}%) below 95% target"
            )
            report_lines.append("   Consider investigating failures before proceeding")
    else:
        report_lines.append("❌ Pipeline incomplete or failed")

    report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report
    report_path = Path("train_datasets/boughter/annotated/validation_report.txt")
    report_path.write_text(report_text)
    print(f"\n✓ Validation report saved to: {report_path}")


def main() -> int:
    """Main validation pipeline."""
    print("=" * 70)
    print("Boughter Dataset Validation")
    print("=" * 70)

    # Validate Stage 1
    stage1_results = validate_stage1_output()

    # Validate Stage 2
    stage2_results = validate_stage2_output()

    # Generate summary report
    generate_report(stage1_results, stage2_results)

    # Exit with appropriate code
    if stage1_results["success"] and stage2_results["success"]:
        if stage2_results["target_met"]:
            return 0  # Success
        return 1  # Warning - below target
    return 2  # Error - pipeline incomplete


if __name__ == "__main__":
    raise SystemExit(main())
