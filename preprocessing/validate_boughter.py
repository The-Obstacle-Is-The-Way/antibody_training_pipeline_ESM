#!/usr/bin/env python3
"""
Boughter Dataset Validation Script

Validates the processed Boughter dataset to ensure quality and correctness.

Usage:
    python3 preprocessing/validate_boughter.py

Checks:
1. ANARCI annotation success rate (target: >95%)
2. CDR length distributions vs expected ranges
3. Sequence quality metrics (no stop codons, limited X's)
4. Training set balance
5. Fragment file integrity

Outputs:
    - Console report with validation results
    - test_datasets/boughter/validation_report.txt
"""

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def validate_stage1_output() -> Dict:
    """Validate Stage 1 output (boughter.csv)."""
    print("\n" + "=" * 70)
    print("Stage 1 Validation: DNA Translation & Novo Flagging")
    print("=" * 70)

    csv_path = Path("test_datasets/boughter.csv")

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
    def check_x_ratio(seq):
        if pd.isna(seq):
            return False
        return seq.count("X") / len(seq) > 0.05

    heavy_x = df["heavy_seq"].apply(check_x_ratio).sum()
    light_x = df["light_seq"].apply(check_x_ratio).sum()
    print(f"  Excessive X's (>5%) in heavy: {heavy_x}")
    print(f"  Excessive X's (>5%) in light: {light_x}")

    # Length distribution
    heavy_len = df["heavy_seq"].str.len()
    light_len = df["light_seq"].str.len()
    print(f"\n✓ Sequence Length Ranges:")
    print(
        f"  Heavy: {heavy_len.min()}-{heavy_len.max()} aa "
        f"(mean: {heavy_len.mean():.1f})"
    )
    print(
        f"  Light: {light_len.min()}-{light_len.max()} aa "
        f"(mean: {light_len.mean():.1f})"
    )

    # Check Novo flagging distribution
    print(f"\n✓ Novo Flagging Strategy:")
    for category in ["specific", "mild", "non_specific"]:
        count = len(df[df["flag_category"] == category])
        pct = count / len(df) * 100
        included = len(
            df[(df["flag_category"] == category) & df["include_in_training"]]
        )
        print(
            f"  {category:15s}: {count:4d} ({pct:5.2f}%) - "
            f"{included} in training"
        )

    # Training set balance
    training_df = df[df["include_in_training"]]
    print(f"\n✓ Training Set Balance:")
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


def validate_stage2_output() -> Dict:
    """Validate Stage 2 output (fragment CSVs + annotation)."""
    print("\n" + "=" * 70)
    print("Stage 2 Validation: ANARCI Annotation & Fragment Extraction")
    print("=" * 70)

    # Check if Stage 2 has been run
    output_dir = Path("test_datasets/boughter")
    vh_file = output_dir / "VH_only_boughter.csv"

    if not vh_file.exists():
        return {
            "success": False,
            "error": f"Stage 2 output not found: {vh_file}",
        }

    # Load one fragment to check counts
    # Skip comment lines starting with #
    df_vh = pd.read_csv(vh_file, comment="#")

    # Load Stage 1 input
    df_stage1 = pd.read_csv("test_datasets/boughter.csv")

    stage1_count = len(df_stage1)
    stage2_count = len(df_vh)
    failures = stage1_count - stage2_count
    failure_rate = (failures / stage1_count) * 100
    success_rate = 100 - failure_rate

    print(f"\n✓ ANARCI Annotation Success:")
    print(f"  Stage 1 input:      {stage1_count} sequences")
    print(f"  Stage 2 annotated:  {stage2_count} sequences")
    print(f"  Failures:           {failures} ({failure_rate:.2f}%)")
    print(f"  Success rate:       {success_rate:.2f}%")

    # Check if we met the >95% target
    target_met = success_rate >= 95.0
    status = "✅ PASS" if target_met else "❌ FAIL"
    print(f"  Target (>95%):      {status}")

    # Check failures by subset
    if (output_dir / "annotation_failures.log").exists():
        failures_log = (output_dir / "annotation_failures.log").read_text().strip()
        if failures_log:
            failure_ids = failures_log.split("\n")
            print(f"\n✓ Failures by Subset:")

            from collections import Counter

            subset_counts = Counter()
            for fid in failure_ids:
                if "hiv_nat" in fid:
                    subset_counts["hiv_nat"] += 1
                elif "hiv_cntrl" in fid:
                    subset_counts["hiv_cntrl"] += 1
                elif "hiv_plos" in fid:
                    subset_counts["hiv_plos"] += 1
                elif "gut_hiv" in fid:
                    subset_counts["gut_hiv"] += 1
                elif "mouse_iga" in fid:
                    subset_counts["mouse_iga"] += 1
                elif "flu" in fid:
                    subset_counts["flu"] += 1

            for subset in sorted(subset_counts.keys()):
                subset_total = len(df_stage1[df_stage1["subset"] == subset])
                subset_failures = subset_counts[subset]
                subset_fail_pct = (subset_failures / subset_total) * 100
                print(
                    f"  {subset:12s}: {subset_failures:3d}/{subset_total:3d} "
                    f"({subset_fail_pct:5.1f}%)"
                )

    # Check CDR length distributions
    print(f"\n✓ CDR Length Distributions (from VH_only):")
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
            f"  Status: ❌ Expected {expected_fragments}, "
            f"found {len(fragment_files)}"
        )

    return {
        "success": True,
        "annotated_sequences": stage2_count,
        "failures": failures,
        "success_rate": success_rate,
        "target_met": target_met,
        "fragment_files": len(fragment_files),
    }


def generate_report(stage1_results: Dict, stage2_results: Dict):
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
                f"Annotated sequences:   {stage2_results['annotated_sequences']}",
                f"Annotation failures:   {stage2_results['failures']}",
                f"Success rate:          {stage2_results['success_rate']:.2f}%",
                f"Target (>95%):         {status}",
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
                f"({stage2_results['success_rate']:.2f}%) below 95% target"
            )
            report_lines.append(
                "   Consider investigating failures before proceeding"
            )
    else:
        report_lines.append("❌ Pipeline incomplete or failed")

    report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report
    report_path = Path("test_datasets/boughter/validation_report.txt")
    report_path.write_text(report_text)
    print(f"\n✓ Validation report saved to: {report_path}")


def main():
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
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Warning - below target
    else:
        sys.exit(2)  # Error - pipeline incomplete


if __name__ == "__main__":
    main()
