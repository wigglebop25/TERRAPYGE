# Evaluation Caveats

Known limitations in the masking and cross-validation experiments, with the
evidence that establishes them. Each item states what the experiment reports,
what the evidence shows instead, and what may still be claimed.

Reproduce every number below with:

```bash
python scripts/diagnose_masking_artifact.py
```

The script writes `results/masking_artifact_diagnosis.json` (gitignored, like the
other metrics files).

---

## 1. The XGBoost mask-rate rise is an artifact

**Reported.** In `results/spatial_masking.json`, XGBoost test AUC appears to rise
as more topographic evidence is withheld from the test nodes:

| Model | 10% | 20% | 30% |
|-------|-----|-----|-----|
| Logistic Regression | 0.3946 | 0.3216 | 0.3646 |
| Random Forest | 0.8023 | 0.5345 | 0.4478 |
| XGBoost | 0.5917 | 0.7220 | 0.8670 |

**Status: dismissed as an artifact.** Three lines of evidence.

### 1.1 It is not a test-composition effect

Masked test sets are nested, so a rise could reflect which nodes fall into the
larger subset. Scoring every trained model against every nested test subset
separates the two explanations. AUC is flat across the columns and moves only
down the rows:

| trained at | test@10% | test@20% | test@30% | all masked |
|------------|----------|----------|----------|------------|
| 10% | 0.6051 | 0.6141 | 0.6272 | 0.6368 |
| 20% | 0.6983 | 0.7102 | 0.7086 | 0.7149 |
| 30% | 0.8862 | 0.8694 | 0.8670 | 0.8694 |

The driver is the training set, not the test population.

### 1.2 It does not survive independent seeds

Re-running the whole experiment under five permutation seeds, the monotonic rise
appears in only **2 of 5**. The seed-to-seed spread is larger than any
between-rate difference:

| perm seed | 10% | 20% | 30% | monotonic |
|-----------|-----|-----|-----|-----------|
| 42 | 0.6051 | 0.7102 | 0.8670 | yes |
| 0 | 0.7799 | 0.5035 | 0.7981 | no |
| 1 | 0.8448 | 0.7235 | 0.8492 | no |
| 2 | 0.7131 | 0.5000 | 0.5663 | no |
| 3 | 0.6651 | 0.6856 | 0.8507 | yes |
| mean | 0.7216 | 0.6246 | 0.7863 | |
| std | 0.0842 | 0.1010 | 0.1124 | |

### 1.3 The scores collapse, so AUC measures tie-ordering

XGBoost places **98.2-99.7% of its gain on the masked block**:

| trained at | masked block gain | kept block gain |
|------------|-------------------|-----------------|
| 10% | 0.997 | 0.003 |
| 20% | 0.988 | 0.011 |
| 30% | 0.982 | 0.018 |

`static_fs` and `critical_acc` each reach AUC 1.0000 on their own. Once those
columns are zeroed at test time, every node follows the same path and the output
collapses to a near-constant:

| trained at | score std | distinct values (of 3,989 nodes) |
|------------|-----------|----------------------------------|
| 10% | 0.000019 | 15 |
| 20% | 0.000012 | 18 |
| 30% | 0.000032 | 74 |

AUC over a near-constant score is the ordering of a handful of leaf values, which
is why it moves by ~0.1 between seeds.

**Root cause.** The baselines are trained on all 20 features and then scored with
9 of them zeroed. That train/test feature-distribution mismatch is what the
numbers measure, not the recovery of withheld topographic evidence.

---

## 2. The masked block is not the only route to the label

**Status: open, tracked as Story 2.5.** This is a separate issue from the
artifact above and affects the headline claim, not just one baseline.

The label is a deterministic function of slope (`slope` alone reaches AUC
0.9999). Masking the topographic block therefore only removes the label driver if
no kept feature encodes terrain position. Four of them do, as Spearman rho
against slope shows:

| Kept feature | rho with slope | single-feature AUC |
|--------------|----------------|--------------------|
| chirps | 0.7294 | 0.9523 |
| worldcover | -0.7195 | 0.8948 |
| jrc_water | -0.6541 | 0.6524 |
| dynamicworld | -0.6507 | 0.8213 |

These are spatially smooth fields over the same 13,297 slope units, so they act
as spatial proxies for the masked terrain.

Consequently, an XGBoost trained on the **kept block only** - with no train/test
mismatch - reaches AUC **0.982** (std 0.004 across five seeds) under the same
random-split masking protocol, against **0.969** for the dual-edge GNN.

**What this does and does not show.** It shows that the reported gap between the
GNN and the non-graph baselines under masking is an artifact of the mismatch: the
baselines are penalised for being scored on features they were trained on. It
does **not** by itself prove the graph adds nothing, because the GNN was not
retrained on the kept block, and the 0.982 figure comes from the random-split
protocol rather than the blocked inductive one (0.873). Both gaps need closing
before the masking experiment can support a claim about graph value.

Until then, the masking results should be read as a demonstration that message
passing tolerates missing inputs better than a model that expects them, not as
evidence that the graph recovers information the kept covariates do not already
contain.

---

## 3. Two input features are constant

`era5` and `arias` each have a single unique value across all 13,297 nodes
(n_unique = 1), so 2 of the 20 model inputs carry no information. The
multi-band source rasters are reduced to band 1 during reprojection (see
`DATA_SCHEMA.md`), which is the likely cause. Either the upstream extraction
should select informative bands, or the columns should be dropped so the feature
count reflects the usable inputs.

---

## Consequences for reporting

- Do not cite the XGBoost mask-rate trend as a finding. It is seed noise over a
  collapsed score.
- Do not describe the masked experiment as withholding topographic evidence. It
  withholds nine columns, four kept covariates still encode terrain.
- Keep the blocked inductive result (0.873) as the conservative figure, but treat
  the comparison against baselines as provisional until Story 2.5 is resolved.
