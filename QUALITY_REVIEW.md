# Delivery review

Overall: 4.0 / 5 across five quality axes. The local implementation and its real-data checks are complete; the limits of a historical feature-only corpus remain explicit.

Dashboard update: the suite now passes 25 tests, including real HTTP requests, local process execution and stopping, and a missing-data failure. A full run was also started and opened through the browser. The interface uses plain labels, reads measured reports, and calculates inspected scores from saved weights. Desktop and narrow layouts were checked. The design is deliberately restrained; the depth is in the working controls and evidence rather than decoration.

| Axis | Score | Evidence and remaining limit |
| --- | --- | --- |
| Accuracy | 4 / 5 | 20 real-data tests passed; full CLI run and checkpoint/hash audits passed. Finite optimization does not establish certified robustness or exact C&W boundary distances. |
| Completeness | 4 / 5 | All requested CLI stages and attack/poison/defense families execute. The supplied dataset cannot support temporal, host or campaign isolation, or reconstruct realizable messages. |
| Clarity | 4 / 5 | README documents formulas, calibration, commands, output schemas and assumptions. Its comprehensive experimental detail requires substantial reading. |
| Actionability | 4 / 5 | A working local environment, fitted models, reports and direct workspace commands are supplied. Linux/macOS CI was not executed here. |
| Conciseness | 4 / 5 | Final delivery can be reviewed through the README, verification evidence and generated report. Some limitations recur where they affect distinct experiments. |

No axis scored at or below 2. Useful next research extensions are a corpus with campaign/time identifiers and raw-message feasibility checks, and stronger attack convergence studies. These extensions should retain the same real-data execution and evidence requirements.

Self-check: the assessment should be consistent with the user's request because results, commands and tests are executed, while unavailable metadata and theoretical guarantees are not invented. Verdict: deliver the verified local research system with its measured results and stated scope.

## Expanded dashboard review — 24 November 2026

Overall: 3.8 / 5. The added analysis features run on verified source records and pinned models. Browser acceptance remains incomplete because the Windows browser tool cannot start.

| Axis | Score | Evidence and remaining improvement |
| --- | --- | --- |
| Accuracy | 4 / 5 | Live checks across five runs recalculated 13,385 predictions with zero difference; all training-model inspectors and source-measurement exports passed. Fresh browser-rendered output still needs verification. |
| Completeness | 3 / 5 | Record exploration, comparisons, exports, and detailed training inspection are implemented and tested through real HTTP requests. The new controls have not received a browser click-through because tool startup fails with Windows error 1920. |
| Clarity | 4 / 5 | Plain labels distinguish selected versus changed records, matching test versus attack subsets, and models trained before versus after cleanup. Dense comparison tables need a fresh narrow-screen review. |
| Actionability | 4 / 5 | The expanded dashboard is running locally; README describes the new paths, and scripts/audit_dashboard_views.py repeats the live checks. Browser automation is unavailable in this session. |
| Conciseness | 4 / 5 | Three substantial features share existing run selection and evidence checks. Verification details are kept in local reports; documentation retains historical audit sections for traceability. |

Highest-priority improvement: repeat the browser audit for the new navigation, filters, downloads, modal controls, keyboard paths, and narrow layouts when the browser tool starts successfully. This cannot be replaced by passing server tests. The remaining gap is explicitly reported to the user.

Self-check: the data and model claims have executed evidence. The user would reasonably expect the new browser workflows to be clicked through; this review does not claim that missing pass occurred.

## Decision tools delivery review

Overall: 3.8 / 5. Accuracy 4: 113 full-suite tests, 22 strengthened focused checks, 105 live model experiments, and 1,000 independently matched cross-run score pairs passed; browser output remains unobserved. Completeness 3: decision previews, validation calibration, distributions, CSV exports, paired mistakes, and fresh record inspection are implemented, but browser acceptance remains blocked by Windows error 1920. Clarity 4: partition boundaries, undefined precision, unmatched records, preview-only decisions, rounding, and source units are explicit; the dense tables need a fresh narrow-screen review. Actionability 4: both new pages are served locally and have repeatable live audit scripts; browser automation remains unavailable. Conciseness 4: the new capabilities share the selected run and existing record dialog; historical verification notes retain some repeated context.

The highest-priority remaining improvement is the browser pass for the new controls, keyboard paths, downloads, and layouts when the browser-control process can start. Self-check: the live-data evidence supports the model claims, while the documented browser gap prevents a claim that every visible workflow has passed.

## Input dependence and attack overlap review

Overall: 3.8 / 5. Accuracy 4: 143 full-suite tests, independent actual-model shuffle/overlap checks, and 3,990 live trials passed; the first live run exposed a real inference-batch inconsistency, which was corrected and rechecked without weakening score equality. Completeness 3: both tools, detailed trial views, attack inspectors, pagination, and complete CSV exports are implemented; Windows error 1920 still prevents browser acceptance. Clarity 4: the UI separates already-missed spam from attack escapes, labels undefined overlap shares, exposes negative importance, distinguishes spread from confidence, and explains why shuffling is neither causal attribution nor a feasible attack; the dense tables need fresh viewport review. Actionability 4: the new pages run locally and have repeatable real-data audit scripts; browser automation is unavailable. Conciseness 4: each new page answers a distinct investigative question and reuses existing model selection and dialogs; the growing sidebar needs a future usability pass.

Highest-priority improvement: complete browser input, download, keyboard, and narrow-layout checks when the control process can start. Self-check: the evidence supports the data and inference claims; it does not justify calling the visual workflows fully verified.
