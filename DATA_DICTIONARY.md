# Data dictionary

| Field | Meaning |
|---|---|
| repository_id, repository_name | De-identified repository code R01--R14. |
| pr_id | Synthetic within-repository PR identifier. |
| merged_at | Synthetic timestamp preserving within-repository order only. |
| outcome_proxy | Marker-based, file-overlapping post-merge corrective-activity proxy; not a verified defect label. |
| is_right_censored | Whether the 30-day window is incomplete; all released modeling rows are false. |
| c1_code_churn | Added plus deleted lines in analyzable files. |
| c2_method_complexity | Complexity of parsed functions/methods intersecting changed target-side lines. |
| c3_static_findings | ESLint warnings and errors in the analyzed PR snapshot; not an introduced-warning delta. |
| c4_author_unfamiliarity | Churn-weighted inverse prior file/directory touches; exposure, not comprehension. |
| analyzable_file_count | Eligible source files considered by the analyzer. |
| test_file_count | Test files observed for the PR. |
| analysis_error_file_count | Eligible files with an analyzer error. |
| test_file_ratio | Test-file share. |
| analysis_error_ratio | Analyzer-error share. |
| directory_fallback_ratio | Share using directory-level history fallback. |
| no_history_ratio | Share with no prior file/directory history. |
| source_cohort | Legacy or expansion cohort. |
| split | Locked repository-wise temporal train, validation, or test assignment. |
