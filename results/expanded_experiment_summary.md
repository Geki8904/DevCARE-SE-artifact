# DevCARE-SE expanded experiment summary

- Dataset: 730 eligible PRs, 14 repositories, 375 positive and 355 negative.
- Temporal test: 225 PRs, 90 positive (prevalence 0.400).
- Highest pooled PR-AUC: Hybrid Expert + Logistic = 0.554839.
- C2-only PR-AUC = 0.554820; paired delta = 0.000019, 95% CI [-0.0526, 0.0519].
- Selected point-estimate intervals use 5,000 repository-cluster resamples across 14 repositories and are stored in `selected_pr_auc_repository_cluster_intervals`.
- Hybrid Logistic vs expert delta = 0.0440; PR-level CI [-0.0023, 0.0918], clustered CI [-0.0660, 0.1595].
- C4-only PR-AUC = 0.357387; Logistic Context minus Code CI crosses zero.
- LORO pooled PR-AUC: RF Code = 0.557273; RF Context = 0.556768.
- LORO macro valid-repository PR-AUC: RF Code = 0.681604; RF Context = 0.699577 (12 valid repositories).
- Legacy cohort best: Random Forest Context PR-AUC = 0.671832.
- Expansion cohort best: Expert DevCARE-SE PR-AUC = 0.567002.
- Complete-case sensitivity: 630 PRs from 13 repositories remain after excluding any PR with analyzer errors; Hybrid Logistic PR-AUC = 0.581715 versus C2 = 0.572919 (delta 0.008795, 95% CI [-0.0348, 0.0554]).

## Interpretation

The expanded evidence does not support a universal winner or a stable incremental gain from C4. Hybrid Logistic and C2-only tie in pooled PR-AUC, while C2 performs better at the top-10% queue. Strong cohort and repository heterogeneity requires reporting temporal, macro-repository, LORO, clustered-bootstrap, and cohort-sensitivity results together.
