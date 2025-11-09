# Novo 2025 Full Replication Plan

_Last updated: 2025-11-08_  
_Source papers: Sakhnini et al. 2025 main manuscript + novo-media-1 supplement_

## Purpose
- Capture the exact scope of the Novo Nordisk study so we stop guessing about the baseline.
- Document what the current repository already satisfies (PLM/ESM track) vs. what is missing (descriptor track + validation).
- Provide a concrete implementation plan, including the complete descriptor inventory, so we can open a GitHub issue or execution epic without ambiguity.

## Paper Baseline (What We Must Match)
### Shared Data Preparation
- Input dataset: Boughter et al. (2020) ELISA panel (human + mouse IgA antibodies).
- Label policy: 0 flags → **specific (class 0)**, >3 flags → **poly-reactive (class 1)**, 1–3 flags are discarded.
- Sequence handling:
  - Annotate sequences with ANARCI using IMGT numbering.
  - Assemble 16 fragments (VH, VL, per-CDR, joined CDRs/FWRs, VH+VL, etc.) as in Table 4 of the paper.

### Track A – PLM Logistic Regression (Already in Repo)
- Embeddings: ESM-1v (mean pooling of final-layer token vectors). Other PLMs were evaluated but ESM-1v VH LogisticReg was the winner.
- Classifier: Scikit-learn LogisticRegression (no hidden layers) per fragment, with VH mean-mode reported as best (10-fold CV ≈ 71% accuracy).
- Validation: 3/5/10-fold CV + Leave-One-Family-Out + external Jain dataset (0 vs >3 flags) + qualitative checks on Shehata/Harvey.

### Track B – Biophysical Descriptor Logistic Regression (Missing Today)
- Feature set: 68 sequence-derived descriptors (Table S1) covering aggregation, flexibility, HPLC retention, multiple hydrophobicity scales, polarity, disorder, charge, etc. Only charge@pH6, charge@pH7.4, and theoretical pI are computed via Biopython; the rest are Schrödinger BioLuminate descriptors.
- Modeling workflow:
  1. Compute all 68 descriptors for VH fragments.
  2. Train LogisticReg on all descriptors; rank features by absolute coefficients.
  3. Select top 25 features; run permutation importance, single-descriptor models, and leave-one-feature-out analysis.
  4. Exhaustive search over combinations of the top descriptors (2/3/4/5) and PCA variants (top 3/5/10 components).
  5. Report that theoretical pI dominates performance and appears in every high-performing subset.

### Validation Obligations
- Every model (PLM and descriptor) must report accuracy, sensitivity, specificity for 3/5/10-fold CV and Leave-One-Family-Out.
- External validation on Jain (ELISA assay) with the same parsing. Shehata & Harvey are used to show distribution shifts (PSR assay) even if accuracy is not meaningful.
- Figures/Tables to replicate: Figure 1 (overall workflow + CV bars), Figure 2 (descriptor importance & comparisons), Table 1 (top descriptor combos), Table S1/S2 (descriptor definitions & coefficients).

## Current Repository Coverage vs Paper
| Capability | Files / Evidence | Status |
|------------|------------------|--------|
| Boughter parsing + VH pipeline | `preprocessing/boughter/*`, `src/antibody_training_esm/datasets/boughter.py` | ✅ Implemented |
| ESM-1v embedding + caching | `src/antibody_training_esm/core/embeddings.py`, `core/trainer.py:63-179` | ✅ Implemented |
| LogisticReg training & CV | `src/antibody_training_esm/core/trainer.py:181-328` | ✅ Implemented |
| Descriptor feature computation | _None_ (README still lists it as “To-Be Implemented”, `README.md:59`) | ❌ Missing |
| Descriptor LogisticReg models | _None_ | ❌ Missing |
| Descriptor-specific plots/analysis | _None_ | ❌ Missing |
| External dataset prep (Jain/Shehata/Harvey) | `src/antibody_training_esm/datasets/{jain,shehata,harvey}.py` | 🔶 Data loaders exist, but descriptor features are not attached |
| Documentation of descriptor baseline | only `docs/other/NOVO_TRAINING_METHODOLOGY.md` narrative | 🔶 Needs actionable plan |

## Implementation Plan to Reach Full Parity
1. **Descriptor Feature Engine**
   - Implement a feature generator module (e.g., `antibody_training_esm/features/biophysical.py`).
   - For Schrödinger descriptors: decide whether to call BioLuminate via command line, cached CSV exports, or reimplement open-source approximations; document licensing constraints.
   - For Biopython descriptors (charge at pH 6/7.4 + theoretical pI): use `Bio.SeqUtils.ProtParam.ProteinAnalysis` with Novo’s PH assumptions.
   - Persist descriptor matrices per dataset split (e.g., `train_datasets/boughter/descriptors/vh_top25.parquet`).

2. **Dataset Integration**
   - Extend `load_data` to return `(embeddings, descriptors, labels)` or parallel loaders so Track A and Track B can share caching logic without intermixing features.
   - Version descriptor caches by hash (analogous to `get_or_create_embeddings`) to keep VH feature matrices reproducible.

3. **Descriptor Model Training**
   - Mirror `BinaryClassifier` with a descriptor-specific pipeline (standardization + LogisticRegression with the same solver parameters used in PLM track).
   - Implement utilities to rank coefficients, run permutation importance, and evaluate single-feature + leave-one-feature-out accuracy exactly as described in Section 2.5.
   - Support PCA-based baselines and exhaustive enumeration for the top 2/3/4/5 descriptor combos.

4. **Evaluation + Reporting**
   - Reuse existing CV harness to emit the same metrics for descriptor models (3/5/10-fold CV, Leave-One-Family-Out, Jain external).
   - Produce summary artifacts matching Figure 2 (coefficients, permutation importance, single-descriptor accuracy, leave-one-out drop, correlation heatmap).
   - Update `docs/` with side-by-side performance tables (ESM vs descriptors) and highlight theoretical pI’s contribution (Table 1 style).

5. **Automation & CI**
   - Add Make/CLI targets (e.g., `make descriptors`, `cli/train.py --track descriptors`).
   - Include descriptor regression tests (unit tests that verify computed pI, charge, hydrophobicity against known sequences).
   - Extend CI matrices so descriptor jobs run alongside PLM jobs once feature generation is deterministic.

6. **Documentation / Issue Template**
   - Create a GitHub issue or project card summarizing the above tasks with owners, ETA, and blockers (e.g., Schrödinger licensing).
   - Update README + USAGE to describe both tracks once implemented.

## Descriptor Inventory (Exact Table S1)
_All descriptors are sourced from Schrödinger BioLuminate unless noted as Biopython. This is the authoritative list the Novo team used to train the VH descriptor models._

| # | Descriptor | Definition | Tool Source |
|---|-----------|------------|-------------|
| 1 | AGGRESCAN_Nr_hotspots | Number of aggregation hotspots computed by the Aggrescan algorithm (http://bioinf.uab.es/aap/aap_help.html) | Schrödinger BioLuminate |
| 2 | Aa_Composition | The total of amino acid composition as described by McCaldon and Argos (Proteins: Structure, Function and Genetics 4:99-122(1988)) | Schrödinger BioLuminate |
| 3 | Aa_Composition_Swissprot | The total value of amino acid composition based on SwissProt annotation (Release notes for UniProtKB/Swiss-Prot release 2013_04 - April 2013) | Schrödinger BioLuminate |
| 4 | Aa_Flexibility_VTR | The total amino acid flexibility as defined by Vihinen, Torkkila, and Rikonen (https://www.ncbi.nlm.nih.gov/pubmed/8090708) | Schrödinger BioLuminate |
| 5 | Aggrescan_av4 | a4v values over a sliding window, as determined by the Aggrescan algorithm | Schrödinger BioLuminate |
| 6 | Aggrescan_av4_pos | a4v positive values over a sliding window, as determined by the Aggrescan algorithm | Schrödinger BioLuminate |
| 7 | All_Aggrescan_a4v_pos | The sum of the average of a4v positive values over a sliding window, as determined by the Aggrescan algorithm | Schrödinger BioLuminate |
| 8 | Alpha_Helix_Chou_Fasman | Alpha helix propensity, as defined by Chou and Fasman (Adv. Enzym. 47:45-148(1978)) | Schrödinger BioLuminate |
| 9 | Alpha_Helix_Deleage_Roux | Alpha helix propensity, as defined by Deleage and Roux (Protein Engineering 1:289-294(1987)) | Schrödinger BioLuminate |
| 10 | Alpha_Helix_Levitt | Alpha helix propensity, as defined by Levitt (Biochemistry 17:4277-4285(1978)) | Schrödinger BioLuminate |
| 11 | Antiparallel_Beta_Strand | Antiparallel beta strand propensity, as defined by Lifson and Sander (Nature 282:109-111(1979)) | Schrödinger BioLuminate |
| 12 | Average_Flexibility_BP | Total amino acid flexibility, as defined by Bhaskaran and Ponnusamy (Int. J. Pept. Protein. Res. 32:242-255(1988)) | Schrödinger BioLuminate |
| 13 | Avg_Area_Buried | Average standard-state to folded-protein buried area, as defined by Rose et al. (Science 229:834- 838(1985)) | Schrödinger BioLuminate |
| 14 | Beta_Sheet_Chou_Fasman | Beta sheet propensity, as defined by Chou and Fasman (Adv. Enzym. 47:45-148(1978)) | Schrödinger BioLuminate |
| 15 | Beta_Sheet_Deleage_Roux | Beta sheet propensity, as defined by Deleage and Roux (Protein Engineering 1:289-294(1987)) | Schrödinger BioLuminate |
| 16 | Beta_Sheet_Levitt | Beta sheet propensity, as defined by Levitt (Biochemistry 17:4277-4285(1978)) | Schrödinger BioLuminate |
| 17 | Beta_Turn_Chou_Fasman | Beta turn propensity, as defined by Chou and Fasman (Adv. Enzym. 47:45-148(1978)) | Schrödinger BioLuminate |
| 18 | Beta_Turn_Deleage_Roux | Beta turn propensity, as defined by Deleage and Roux (Protein Engineering 1:289-294(1987)) | Schrödinger BioLuminate |
| 19 | Beta_Turn_Levitt | Beta turn propensity, as defined by Levitt (Biochemistry 17:4277-4285(1978)) | Schrödinger BioLuminate |
| 20 | Bulkiness | Total amino acid bulkiness (J. Theor. Biol. 21:170-201(1968)) | Schrödinger BioLuminate |
| 21 | Charge at pH 6 | Charge of the protein at pH 6 | Biopython ProteinAnalysis |
| 22 | Charge at pH 7.4 | Charge of the protein at pH 7.4 | Biopython ProteinAnalysis |
| 23 | Coil_Deleage_Roux | Total score for coil, as defined by Deleage and Roux (Protein Engineering 1:289-294(1987)) | Schrödinger BioLuminate |
| 24 | Disorder_Propensity_DisProt | Total disorder promotion propensity (https://www.ncbi.nlm.nih.gov/pubmed/17578581) | Schrödinger BioLuminate |
| 25 | Disorder_Propensity_FoldUnfold | Total disorder promotion propensity (https://www.ncbi.nlm.nih.gov/pubmed/15498936) | Schrödinger BioLuminate |
| 26 | Disorder_Propensity_TOP_IDP | Total disorder propensity for intrinsic disorder, based on the TOP-IDP scale model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2676888/) | Schrödinger BioLuminate |
| 27 | HPLC_Retention_Ph_2_1 | Total value of retention coefficients in HPLC at pH 2.1 (Proc. Natl. Acad. Sci. USA 77:1632- 1636(1980)) | Schrödinger BioLuminate |
| 28 | HPLC_Tfa_Retention | Total value of retention coefficients in HPLC/TFA (Anal. Biochem. 124:201-208(1982)) | Schrödinger BioLuminate |
| 29 | Hplc_Hfba_Retention | Total value of retention coefficients in HFBA (Anal. Biochem. 124:201-208(1982)) | Schrödinger BioLuminate |
| 30 | Hplc_Retention_Ph_7_4 | Total value of retention coefficients in HPLC at pH 7.4 (Proc. Natl. Acad. Sci. USA 77:1632- 1636(1980)) | Schrödinger BioLuminate |
| 31 | Hydrophobicity_Abraham_Leo | Total hydrophobicity, as defined by Abraham and Leo (Proteins: Structure, Function and Genetics 2:130-152(1987)) | Schrödinger BioLuminate |
| 32 | Hydrophobicity_Black | Total hydrophobicity, as definedy by Black (Anal. Biochem. 193:72-82(1991)) | Schrödinger BioLuminate |
| 33 | Hydrophobicity_Bull_Breese | Total hydrophobicity, as definedy by Bull and Breese (Arch. Biochem. Biophys. 161:665- 670(1974)) | Schrödinger BioLuminate |
| 34 | Hydrophobicity_Chothia | Total hydrophobicity based on proportion of buried residues (95%), as defined by Chothia (J. Mol. Biol. 105:1-14(1976)) | Schrödinger BioLuminate |
| 35 | Hydrophobicity_Eisenberg | Total normalized consensus hydrophobicity, as defined by Eisenberg et al. (J. Mol. Biol. 179:125- 142(1984)) | Schrödinger BioLuminate |
| 36 | Hydrophobicity_Fauchere | Total hydrophobicity, as defined by Fauchere (Eur. J. Med. Chem. 18:369-375(1983)) | Schrödinger BioLuminate |
| 37 | Hydrophobicity_Guy | Total hydrophobicity based on free energy of transfer, as defined by Guy (Biophys J. 47:61- 70(1985)) | Schrödinger BioLuminate |
| 38 | Hydrophobicity_Hopp_Woods | Total hydrophilicity, as defined by Hopp & Woods (Proc. Natl. Acad. Sci. U.S.A. 78:3824- 3828(1981)) | Schrödinger BioLuminate |
| 39 | Hydrophobicity_Hplc_Parker | Total hydrophilicity derived from HPLC peptide retention times, as defined by Parker et al. (Biochemistry 25:5425-5431(1986)) | Schrödinger BioLuminate |
| 40 | Hydrophobicity_Hplc_Ph_3_4_Cowan | Total hydrophobicity determined by HPLC at ph 3.4 , as defined by Cowan and Whittaker (Peptide Research 3:75-80(1990)) | Schrödinger BioLuminate |
| 41 | Hydrophobicity_Hplc_Ph_7_5_Cowan | Total hydrophobicity determined by HPLC at ph 7.5, as defined by Cowan and Whittaker (Peptide Research 3:75-80(1990)) | Schrödinger BioLuminate |
| 42 | Hydrophobicity_Hplc_Wilson | Total hydrophobicity derived from HPLC peptide retention times, as defined by Wilson et al. (Biochem. J. 199:31-41(1981)) | Schrödinger BioLuminate |
| 43 | Hydrophobicity_Janin | Total hydrophobicity based on dG of transfer from inside to outside of a globular protein, as defined by Janin (Nature 277:491-492(1979)) | Schrödinger BioLuminate |
| 44 | Hydrophobicity_Kyte_Doolittle | Total hydrophobicity, as defined by Kyte and Dolittle (J. Mol. Biol. 157:105-132(1982)) | Schrödinger BioLuminate |
| 45 | Hydrophobicity_Manavalan | Total average surrounding hydrophobicity, as defined by Manavalan and Ponnusamy (Nature 275:673-674(1978)) | Schrödinger BioLuminate |
| 46 | Hydrophobicity_Miyazawa_Jernigan | Hydrophobicity, as defined by Miyazawa and Jernigan (Macromolecules 18:534-552(1985)) | Schrödinger BioLuminate |
| 47 | Hydrophobicity_Rao_Argos | Total transmembrane helix parameters, as defined by Rao and Argos (Biochim. Biophys. Acta 869:197-214(1986)) | Schrödinger BioLuminate |
| 48 | Hydrophobicity_Rf_Mobility | Total hydrophobicity based on chromatograohic mobility, as defined by Aboderin (Int. J. Biochem. 2:537-544(1971)) | Schrödinger BioLuminate |
| 49 | Hydrophobicity_Rose | Total hydrophobicity based on mean fractional exposed area loss (average area buried/standard state area), as defined by Rose (Science 229:834-838(1985)) | Schrödinger BioLuminate |
| 50 | Hydrophobicity_Roseman | Total hydrophobicity, as defined by Roseman (J. Mol. Biol. 200:513-522(1988)) | Schrödinger BioLuminate |
| 51 | Hydrophobicity_Sweet | Total optimized matching hydrophobicity, as defined by Sweet (J. Mol. Biol. 171:479-488(1983)) | Schrödinger BioLuminate |
| 52 | Hydrophobicity_Tanford | Total hydrophobicity, as defined by Tanford (J. Am. Chem. Soc. 84:4240-4274(1962)) | Schrödinger BioLuminate |
| 53 | Hydrophobicity_Welling | Total antigenicity, as defined by Welling (FEBS Lett. 188:215-218(1985)) | Schrödinger BioLuminate |
| 54 | Hydrophobicity_Wolfenden | Total hydration potential at 25 °C, as defined by Wolfenden (Biochemistry 20:849-855(1981)) | Schrödinger BioLuminate |
| 55 | Molecular_Weight | Molecular weight based on the sum of each amino acid molecular weight | Schrödinger BioLuminate |
| 56 | Number_Of_Codons | Number of codons encoding each amino acid in the universal genetic code | Schrödinger BioLuminate |
| 57 | Parallel_Beta_Strand | Parallel beta strand propensity, as defined by Lifson and Sander (Nature 282:109-111(1979)) | Schrödinger BioLuminate |
| 58 | Percentage_Accessible_Res | Total molar fraction of accessible residues, as defined by Janin (Nature 277:491-492(1979)) | Schrödinger BioLuminate |
| 59 | Percentage_Buried_Res | Total molar fraction of buried residues, as defined by Janin (Nature 277:491-492(1979)) | Schrödinger BioLuminate |
| 60 | Polarity_Grantham | Total polarity, as defined by Grantham (Science 185:862-864(1974)) | Schrödinger BioLuminate |
| 61 | Polarity_Zimmerman | Total polarity, as defined by Zimmerman (J. Theor. Biol. 21:170-201(1968)) | Schrödinger BioLuminate |
| 62 | Ratio_Hetero_End_Side | Total atomic weight ratio of hetero elements in end group to C in side chain (Science 185:862- 864(1974)) | Schrödinger BioLuminate |
| 63 | Recognition_Factors | Total recognition factor of each amino acid, as defined by Fraga (Can. J. Chem. 60:2606-2610(1982)) | Schrödinger BioLuminate |
| 64 | Refractivity | Total refractivity index of each amino acid, as defined by Jones (J. Theor. Biol. 50:167- 184(1975)) | Schrödinger BioLuminate |
| 65 | Relative_Mutability | Total relative mutability (Ala=100), as defined by Dayhoff et al. (In "Atlas of Protein Sequence and Structure", Vol.5, Suppl.3 (1978)) | Schrödinger BioLuminate |
| 66 | Theoretical pI | Isoelectric point | Biopython ProteinAnalysis |
| 67 | Total_Beta_Strand | Total (antiparallel+parallel) beta strand propensity, as defined by Lifson and Sander (Nature 282:109-111(1979)) | Schrödinger BioLuminate |
| 68 | Transmembrane_Tendency | Total transmembrane tendency, as defined by Zhao and London (Protein Sci. 15:1987- 2001(2006)) | Schrödinger BioLuminate |

## Immediate Next Steps
1. Decide how we will compute Schrödinger descriptors (license, cached exports, or open-source analogues). Document the decision before coding.
2. Stand up the feature generator prototype on VH sequences only (matches Novo focus and keeps scope tight).
3. Once descriptors are cached, implement the LogisticReg baseline + analyses (coeff ranking, permutation tests, top-k combos) so we can reproduce Figure 2/Table 1 metrics alongside our existing ESM results.
4. Capture the plan above in a GitHub issue/Project ticket and link this document as the single source of truth.
