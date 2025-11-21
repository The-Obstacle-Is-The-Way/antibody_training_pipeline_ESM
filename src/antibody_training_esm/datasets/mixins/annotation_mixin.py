import logging

import pandas as pd


class AnnotationMixin:
    """Mixin for ANARCI sequence annotation capabilities."""

    logger: logging.Logger

    def annotate_sequence(
        self, sequence_id: str, sequence: str, chain: str
    ) -> dict[str, str] | None:
        """
        Annotate a single sequence using ANARCI (IMGT numbering).

        This method wraps riot_na.create_riot_aa() to extract CDR/FWR regions.

        Args:
            sequence_id: Unique identifier for the sequence
            sequence: Protein sequence to annotate
            chain: Chain type ("H" for heavy, "L" for light)

        Returns:
            Dictionary with keys: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4
            Returns None if annotation fails
        """
        try:
            # Import riot_na here to avoid dependency issues
            from riot_na import create_riot_aa

            # Run ANARCI annotation
            result = create_riot_aa(sequence_id, sequence, chain=chain)

            if result is None:
                self.logger.warning(
                    f"ANARCI annotation failed for {sequence_id} ({chain} chain)"
                )
                return None

            # Extract regions
            annotations = {
                "FWR1": result.get("FWR1", ""),
                "CDR1": result.get("CDR1", ""),
                "FWR2": result.get("FWR2", ""),
                "CDR2": result.get("CDR2", ""),
                "FWR3": result.get("FWR3", ""),
                "CDR3": result.get("CDR3", ""),
                "FWR4": result.get("FWR4", ""),
            }

            # Validate annotations (should not be empty)
            if not any(annotations.values()):
                self.logger.warning(
                    f"All annotations empty for {sequence_id} ({chain} chain)"
                )
                return None

            return annotations

        except Exception as e:
            self.logger.error(f"Error annotating {sequence_id} ({chain} chain): {e}")
            return None

    def annotate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate all sequences in a DataFrame.

        Adds annotation columns for heavy and light chains.

        Args:
            df: DataFrame with VH_sequence and optionally VL_sequence columns

        Returns:
            DataFrame with annotation columns added
        """
        self.logger.info(f"Annotating {len(df)} sequences...")

        # Annotate heavy chains
        if "VH_sequence" in df.columns:
            self.logger.info("Annotating VH sequences...")
            vh_annotations: pd.Series = df.apply(
                lambda row: self.annotate_sequence(
                    row.get("id", f"seq_{row.name}"), row["VH_sequence"], "H"
                )
                if pd.notna(row["VH_sequence"])
                else None,
                axis=1,
            )

            # Extract annotation fields
            for field in ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]:
                df[f"VH_{field}"] = vh_annotations.apply(
                    lambda x, f=field: x.get(f, "") if x else ""
                )

        # Annotate light chains (if present)
        if "VL_sequence" in df.columns:
            self.logger.info("Annotating VL sequences...")
            vl_annotations: pd.Series = df.apply(
                lambda row: self.annotate_sequence(
                    row.get("id", f"seq_{row.name}"), row["VL_sequence"], "L"
                )
                if pd.notna(row["VL_sequence"])
                else None,
                axis=1,
            )

            # Extract annotation fields
            for field in ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]:
                df[f"VL_{field}"] = vl_annotations.apply(
                    lambda x, f=field: x.get(f, "") if x else ""
                )

        self.logger.info("Annotation complete")
        return df
