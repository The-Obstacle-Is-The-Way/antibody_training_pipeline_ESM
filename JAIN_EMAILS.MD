Dear Dr. Xu, Dr. Vasquez, and Prof. Wittrup,

I’m collaborating with Hugging Face Science on an open-source antibody developability pipeline (https://github.com/ludocomito/antibody_training_pipeline_ESM) and am replicating the non-specificity labeling used in Sakhnini et al. (Novo/Cambridge, 2025). 

They cite Jain et al. (PNAS 2017) with ELISA-based flags from a 6-ligand panel (ssDNA, dsDNA, insulin, LPS, cardiolipin, KLH).

The PNAS SI (Dataset S3) provides an aggregate ELISA value but not the per-ligand measurements. The Novo/Cambridge authors mentioned they obtained the per-ligand flags directly from your team.

Would you be willing to share either of the following for the 137 IgG1 antibodies?

The per-ligand ELISA OD values (six columns) and any plate normalization/cutoff criteria you recommend; or

A processed table mapping antibody ID to the six binary flags (and/or total flag count).

We’ll use the data solely to reproduce the published labeling and will cite you explicitly; we won’t redistribute without permission (happy to sign an MTA). If helpful, I can attach our current jain.csv (137 sequences + aggregate assays) so you can confirm the ID mapping.

Thank you for considering this and for the original dataset.





Hello, 

Just FYI, I believe I have reversed engineered the Jain dataset here: 
https://www.biorxiv.org/content/10.1101/2025.04.28.650927v1

Novo Nordisk Parity Attempt - Jain Test Set
Results on 86-Antibody Parity Set:
- Accuracy: 66.28% (57/86) - EXACTLY matches Novo's 66.28%
- Confusion Matrix: [[40, 19], [10, 17]] - Cell-for-cell identical to Novo 

The 8 Removed Antibodies (All SPECIFIC, label=0)
STEP 1: Initial QC (3 VH Length Outliers)
1. crenezumab - VH=112 aa (z=-2.29, extremely short, CDR H3 only 3 residues)
2. fletikumab - VH=127 aa (z=+2.59, extremely long)
3. secukinumab - VH=127 aa (z=+2.59, extremely long)

STEP 2: Novo Parity (5 Biology + Confidence + Clinical QC)
  4. muromonab (OKT3) - MURINE antibody, WITHDRAWN 2010, severe HAMA response, borderline
  SMP/OVA
  5. cetuximab - CHIMERIC antibody, higher immunogenicity, hypersensitivity reactions (3-5%)
  6. girentuximab - CHIMERIC antibody, DISCONTINUED (failed Phase 3 ARISER trial)
  7. tabalumab - HUMAN antibody, DISCONTINUED by Eli Lilly (failed Phase 3 lupus trials)
  8. abituzumab - HUMANIZED antibody, Failed Phase 3 POSEIDON primary endpoint, borderline SMP=0.167




Dear Dr., please find enclosed a table with the per-ligand ELISA data from our 2017 paper.  I believe we sent the same data, maybe the same file, to the authors of the Novo-Cambridge preprint a while back when they made a similar request.  Please let us know if you have any questions.

Also, keep in mind that we made the samples corresponding to the 137 antibodies using fixed human constant regions: IgG1 for the heavy chains and human kappa or lambda depending on the class of the light chain variable region.  For practical reasons we did not attempt to reproduce the entire sequences of the original antibodies.  For example, while muromonab is a “fully murine” antibody, the sample we made and assayed was a chimeric antibody.  This fact was explained in the original publication, and should be considered when comparing to properties of the actual clinical antibodies.

Best,
Max




