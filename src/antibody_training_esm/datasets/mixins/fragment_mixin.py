"""
Mixin for fragment handling (statistics, CSV export).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd


class FragmentMixin:
    """Mixin for antibody fragment generation."""

    logger: logging.Logger
    output_dir: Path
    dataset_name: str

    def get_fragment_types(self) -> list[str]:
        """Expected to be implemented by the main class."""
        raise NotImplementedError

    def create_fragments(self, row: pd.Series) -> dict[str, tuple[str, int, str]]:
        """
        Create all fragment types from an annotated sequence row.

        Args:
            row: DataFrame row with annotation columns

        Returns:
            Dictionary mapping fragment_type -> (sequence, label, source)

        Raises:
            ValueError: If required annotation columns are missing
        """
        fragments = {}
        sequence_id = row.get("id", f"seq_{row.name}")
        label = row.get("label", 0)

        fragment_types = self.get_fragment_types()

        # Validate that required columns exist for requested fragments
        required_cols = set()
        if (
            any(
                ft in fragment_types
                for ft in [
                    "VH_only",
                    "VH+VL",
                    "H-CDR1",
                    "H-CDR2",
                    "H-CDR3",
                    "H-CDRs",
                    "H-FWRs",
                ]
            )
            and "VH_sequence" not in row
        ):
            required_cols.add("VH_sequence")
        if (
            any(
                ft in fragment_types
                for ft in [
                    "VL_only",
                    "VH+VL",
                    "L-CDR1",
                    "L-CDR2",
                    "L-CDR3",
                    "L-CDRs",
                    "L-FWRs",
                ]
            )
            and "VL_sequence" not in row
        ):
            required_cols.add("VL_sequence")

        if required_cols:
            raise ValueError(
                f"Missing required columns for fragment extraction: {sorted(required_cols)}. "
                f"Available columns: {sorted(row.index.tolist())}. "
                "Did annotation fail?"
            )

        # Helper to concatenate regions
        def concat(*regions: Any) -> str:
            return "".join(str(r) for r in regions if pd.notna(r) and r != "")

        # Full antibody fragments
        if "VH_only" in fragment_types:
            fragments["VH_only"] = (row.get("VH_sequence", ""), label, sequence_id)

        if "VL_only" in fragment_types:
            fragments["VL_only"] = (row.get("VL_sequence", ""), label, sequence_id)

        if "VH+VL" in fragment_types:
            vh = row.get("VH_sequence", "")
            vl = row.get("VL_sequence", "")
            fragments["VH+VL"] = (concat(vh, vl), label, sequence_id)

        # Heavy chain fragments
        if "H-CDR1" in fragment_types:
            fragments["H-CDR1"] = (row.get("VH_CDR1", ""), label, sequence_id)
        if "H-CDR2" in fragment_types:
            fragments["H-CDR2"] = (row.get("VH_CDR2", ""), label, sequence_id)
        if "H-CDR3" in fragment_types:
            fragments["H-CDR3"] = (row.get("VH_CDR3", ""), label, sequence_id)

        if "H-CDRs" in fragment_types:
            h_cdrs = concat(
                row.get("VH_CDR1", ""),
                row.get("VH_CDR2", ""),
                row.get("VH_CDR3", ""),
            )
            fragments["H-CDRs"] = (h_cdrs, label, sequence_id)

        if "H-FWRs" in fragment_types:
            h_fwrs = concat(
                row.get("VH_FWR1", ""),
                row.get("VH_FWR2", ""),
                row.get("VH_FWR3", ""),
                row.get("VH_FWR4", ""),
            )
            fragments["H-FWRs"] = (h_fwrs, label, sequence_id)

        # Light chain fragments
        if "L-CDR1" in fragment_types:
            fragments["L-CDR1"] = (row.get("VL_CDR1", ""), label, sequence_id)
        if "L-CDR2" in fragment_types:
            fragments["L-CDR2"] = (row.get("VL_CDR2", ""), label, sequence_id)
        if "L-CDR3" in fragment_types:
            fragments["L-CDR3"] = (row.get("VL_CDR3", ""), label, sequence_id)

        if "L-CDRs" in fragment_types:
            l_cdrs = concat(
                row.get("VL_CDR1", ""),
                row.get("VL_CDR2", ""),
                row.get("VL_CDR3", ""),
            )
            fragments["L-CDRs"] = (l_cdrs, label, sequence_id)

        if "L-FWRs" in fragment_types:
            l_fwrs = concat(
                row.get("VL_FWR1", ""),
                row.get("VL_FWR2", ""),
                row.get("VL_FWR3", ""),
                row.get("VL_FWR4", ""),
            )
            fragments["L-FWRs"] = (l_fwrs, label, sequence_id)

        # Combined fragments
        if "All-CDRs" in fragment_types:
            all_cdrs = concat(
                row.get("VH_CDR1", ""),
                row.get("VH_CDR2", ""),
                row.get("VH_CDR3", ""),
                row.get("VL_CDR1", ""),
                row.get("VL_CDR2", ""),
                row.get("VL_CDR3", ""),
            )
            fragments["All-CDRs"] = (all_cdrs, label, sequence_id)

        if "All-FWRs" in fragment_types:
            all_fwrs = concat(
                row.get("VH_FWR1", ""),
                row.get("VH_FWR2", ""),
                row.get("VH_FWR3", ""),
                row.get("VH_FWR4", ""),
                row.get("VL_FWR1", ""),
                row.get("VL_FWR2", ""),
                row.get("VL_FWR3", ""),
                row.get("VL_FWR4", ""),
            )
            fragments["All-FWRs"] = (all_fwrs, label, sequence_id)

        if "Full" in fragment_types:
            full = concat(
                row.get("VH_sequence", ""),
                row.get("VL_sequence", ""),
            )
            fragments["Full"] = (full, label, sequence_id)

        # Nanobody-specific (VHH)
        if "VHH_only" in fragment_types:
            fragments["VHH_only"] = (row.get("VH_sequence", ""), label, sequence_id)

        return fragments

    def create_fragment_csvs(self, df: pd.DataFrame, suffix: str = "") -> None:
        """
        Generate CSV files for all fragment types.

        Creates one CSV file per fragment type with columns:
        - id: sequence identifier
        - sequence: fragment sequence
        - label: binary label (0=specific, 1=non-specific)
        - source: original sequence ID

        Args:
            df: Annotated DataFrame
            suffix: Optional suffix for output filenames (e.g., "_filtered")
        """
        self.logger.info("Generating fragment CSVs...")

        fragment_types = self.get_fragment_types()

        # Collect fragments for each type
        fragment_data: dict[str, list[dict[str, Any]]] = {
            ftype: [] for ftype in fragment_types
        }

        for _, row in df.iterrows():
            fragments = self.create_fragments(row)
            for ftype, (seq, label, source) in fragments.items():
                if seq:  # Skip empty sequences
                    fragment_data[ftype].append(
                        {
                            "id": f"{source}_{ftype}",
                            "sequence": seq,
                            "label": label,
                            "source": source,
                        }
                    )

        # Write CSV files
        for ftype, data in fragment_data.items():
            if not data:
                self.logger.warning(f"No data for fragment type: {ftype}")
                continue

            output_file = self.output_dir / f"{ftype}_{self.dataset_name}{suffix}.csv"
            fragment_df = pd.DataFrame(data)

            # Write with metadata header
            with open(output_file, "w") as f:
                f.write(f"# Dataset: {self.dataset_name}\n")
                f.write(f"# Fragment type: {ftype}\n")
                f.write(f"# Total sequences: {len(fragment_df)}\n")
                f.write(
                    f"# Label distribution: "
                    f"{(fragment_df['label'] == 0).sum()} specific, "
                    f"{(fragment_df['label'] == 1).sum()} non-specific\n"
                )
                fragment_df.to_csv(f, index=False)

            self.logger.info(
                f"  {ftype}: {len(fragment_df)} sequences → {output_file.name}"
            )

        self.logger.info(f"Fragment CSVs written to {self.output_dir}")
