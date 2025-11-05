# Jain Dataset - Raw Source Files

**DO NOT MODIFY THESE FILES - Original sources only**

---

## Files

### Public Data (Jain et al. 2017 PNAS)

**Citation:** Jain T, Sun T, Durand S, Hall A, Houston NR, Nett JH, Sharkey B, Bobrowicz B, Caffry I, Yu Y, Cao Y, Lynaugh H, Brown M, Baruah H, Gray LT, Krauland EM, Xu Y, Vásquez M, Wittrup KD (2017). "Biophysical properties of the clinical-stage antibody landscape." PNAS 114(5): 944-949.

- `jain-pnas.1616408114.sd01.xlsx` - **Supplementary Data 1: Sequences**
  - VH and VL sequences for 137 clinical-stage antibodies
  - Includes antibody names, targets, clinical stage

- `jain-pnas.1616408114.sd02.xlsx` - **Supplementary Data 2: Biophysical Properties**
  - BVP, PSR, AC-SINS, HIC, CIC, etc.
  - Charge, pI, CDR lengths

- `jain-pnas.1616408114.sd03.xlsx` - **Supplementary Data 3: Thermal Stability**
  - Fab Tm, scFv Tm
  - Results across 12 assays

### Private Data (Adimab)

- `Private_Jain2017_ELISA_indiv.xlsx` - **Private ELISA Data**
  - Individual ELISA reactivity for 6 antigens:
    - Cardiolipin, KLH, LPS, ssDNA, dsDNA, Insulin
  - OD values used to calculate ELISA flags
  - This data is NOT in the published paper
  - **Data source:** Adimab (T. Sun and Y. Xu from the Jain et al. 2017 author list were most responsible for generating this data)
  - **Permission:** Shared with permission from Adimab for research purposes

---

## Conversion to CSV

To convert these Excel files to CSV format:

```bash
python3 scripts/conversion/convert_jain_excel_to_csv.py
```

Output files will be created in `test_datasets/jain/processed/`.

---

## Data Provenance

- **Public data:** Downloaded from PNAS supplementary materials
- **Private ELISA data:** Provided by Adimab (T. Sun and Y. Xu), shared with permission for research purposes
- **Date added:** 2025-11-03
- **Last verified:** 2025-11-05

---

**See:** `JAIN_COMPLETE_GUIDE.md` (repo root) for complete documentation
