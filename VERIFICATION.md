# Executed verification

Verified locally on 23 November 2026 using Python 3.11.9, Windows x64, CPU, and the package versions in `requirements-lock.txt`.

The first section records the original command-line verification. The dashboard additions have a separate verification record below.

The installation command `.venv\Scripts\python.exe -m pip install -e .` completed successfully. `python -m pip check` reported no broken requirements. All 19 Python source and test files parsed successfully.

Final executed checks:

```text
python -m pytest
20 passed in 10.70s

ml-adv-sec verify-all
Running prepare
Running train
Running attack-evasion
Running attack-poison
Running defend
Running evaluate
reports\benchmark_results.md
Verification completed on checksum-verified real data.
Exit code: 0
```

The quick configuration also completed its full real-data pipeline. All stages use actual Spambase records, real model fits, and real ART/CleverHans attack calls. Tests do not mock any library or generate substitute datasets.

The full configuration processed 4,601 authentic source records into 4,204 distinct, unambiguous feature rows. Three groups with conflicting labels were discarded. Partition sizes were 2,522 train, 841 validation, and 841 test.

The final audit reloaded and verified the latest 28 model checkpoints, their weights and dataset hashes, and their source hashes. All 28 met the configured empirical validation FPR constraint. The reports contain five clean-utility rows, 12 attack/transfer rows, 20 robustness/utility sweep points, and 24 poisoned/sanitized model rows. Every final evasion candidate passed the implemented domain constraints. Verification evidence and report SHA-256 hashes are saved in `reports/verification.json`.

Measured model and attack outputs are in `reports/benchmark_results.md` and its JSON companion. They are generated results, not manually authored metric tables. The clean MLP's measured test FPR exceeded the validation target; the target is not a guarantee on unseen data. An ASR with no initially detected spam is reported as undefined rather than zero.

Execution evidence establishes checked invariants and working commands in this environment. It does not establish exact distance-to-boundary, raw-email realizability, present-day spam coverage, or a production security guarantee. Temporal and campaign isolation are unavailable because this corpus lacks those identifiers. The CI workflow is supplied but was not executed on a remote runner during this task.

## Dashboard verification

The expanded suite passed **25 tests in 24.59 seconds**. The dashboard tests use a real HTTP server, actual saved reports and models, and real child processes. They check saved email changes, recalculated scores, protected run controls, invalid settings, completion of a full pipeline, cancellation, rejection of simultaneous runs, and failure when the dataset is absent. No library behavior is replaced in these tests.

A separate full-dataset experiment named `Dashboard live check` was started using the page's **Start run** button. It completed in approximately 51 seconds with exit code 0 and generated all 28 model variants. Its saved report was opened through **Open results**. The browser checks covered desktop and narrow layouts, saved-run selection, filter comparisons, training changes, per-email inspection and JSON export. There was no page-wide horizontal overflow at the checked desktop and narrow sizes, and no browser console errors were reported.

The dashboard remains local at `http://127.0.0.1:8873`. Its job history, settings, process output and model files remain on disk. Machine-readable evidence is in `reports/dashboard_verification.json`. The expanded tests also exposed and verified a fix for ART's strict bounds check when a small data selection contains a constant measurement.

## Record explorer, comparisons, and training inspection — 24 November 2026

The final full expanded suite passed 79 tests in 136.21 seconds after adding the analysis views. Machine-readable results are saved in `reports/dashboard-expanded-tests.xml`. Tests fit actual models on authentic source records and exercise real HTTP requests. Coverage includes all five original/hardened model score recalculations, changed/cleaned models, all 15 label/outcome combinations, sorting, pagination, source-row search, complete filtered CSV exports, all 57 original measurements, paired run differences, spreadsheet formula protection, and rejection of outdated report hashes or invalid inputs. Training inspection checks all three change methods both before and after cleanup, validates saved masks and record IDs, and independently recalculates all four displayed scores.

A separate check of the running server passed across all five saved runs. It recalculated all 25 original/hardened score vectors from pinned weights with no difference from the saved scores, exported every test record with its original measurements, opened training-record details for every changed/cleaned model, checked same-run zero differences and cross-run comparisons, and rejected outdated evidence. Results are in `reports/dashboard_feature_checks.json`; the repeatable check is `scripts/audit_dashboard_views.py`.

Comparison matching checks both the full test partition and the actual attacked source IDs against the source-data hash. Sharing a full test set does not imply sharing the attacked subset. Cleanup marks belong to the matching cleanup trial; API responses identify that model and whether the inspected scores come from before or after cleanup.

Both browser scripts passed Node syntax checks, and the changes passed `git diff --check`. Browser automation could not start during this update: Windows returned `CreateProcessWithLogonW failed: 1920`. Consequently the new buttons, modal layouts, keyboard paths, and responsive behavior have not received a fresh browser click-through. Earlier browser audit evidence describes the earlier interface only. HTTP and real-model checks are complete; they do not substitute for that remaining browser pass.

## Decision lab, paired mistakes, and held-out record details

The full suite passed **113 tests in 137.63 seconds** after the new pages and endpoints were integrated (`reports/dashboard-all-features-tests.xml`). A subsequent focused run passed **22 tests in 24.85 seconds**, strengthening every displayed validation-boundary count and CSV threshold check and adding two new cases for changed-data and cleaned-model calibration (`reports/dashboard-analysis-focused-tests.xml`). Together these runs cover 115 distinct tests. All model fixtures use actual source records and real training, attack, defense, and evaluation stages.

The live server completed **105 actual decision experiments**: saved/manual/validation-target modes for five original/hardened models and a changed/cleaned model pair across all five saved runs. Every recalculated test vector matched its saved scores exactly. Target mode met the requested empirical validation false-alarm rate. Export checks covered both full validation/test partitions, every disagreement category, actual decision thresholds, source IDs, and report fingerprints. Full record inspections returned 57 source measurements and fresh scores from six pinned models.

Independent cross-run CSV checks loaded actual source-ID and score arrays. They verified all **841** score pairs for two runs sharing a test set, all **159** pairs for different overlapping test partitions, and a third comparison with **zero** shared records. No unmatched source record was paired. Machine-readable live evidence is `reports/dashboard_analysis_checks.json`; repeat with `scripts/audit_dashboard_analysis.py`.

The three browser scripts pass Node syntax checks, and the changes pass `git diff --check`. Browser initialization was retried and still failed with Windows error 1920, before any page observation. The decision-lab forms, preview/export buttons, disagreement selectors and pagination, left/right record buttons, and new inspector views therefore remain unverified through actual browser input. Server/model checks do not establish keyboard, layout, or browser download behavior.

## Input importance and attack overlap

The expanded full suite passed **143 tests in 184.67 seconds** (`reports/dashboard-complete-research-tests.xml`). The 28 added checks independently rerun every column-shuffle trial for all five original/hardened models, check validation records and changed/cleaned models, validate repeat means and sample spread, verify reproducible CSV trial values, and check both attack targets across all record categories. Attack checks independently recompute source-ID intersections, eligibility, per-record escapes, all pairwise counts/shares, complete CSV exports, and table-to-inspector sample mappings. Invalid settings and stale fingerprints are rejected.

The live audit passed on all five saved runs: **35 model/partition checks covering 3,990 actual column-shuffle trials**, all attack categories and exports for both targets, every all-records results page at ten rows per page, and all **60 attack-table inspector links**. This covered 246 target/record combinations, 360 overlap cells, and 30 pages. Original test-score checks matched saved results exactly. Evidence is `reports/dashboard_research_checks.json`; repeat with `scripts/audit_dashboard_research.py`.

The first live audit caught batch-dependent floating-point differences between the attack table and inspector, up to approximately 1.2e-7 in the observed records. The inspector had run an original/changed pair as a two-record batch, while evaluation and overlap used separate complete batches. It now recalculates the complete original and changed batches separately, verifies both full vectors, and only then selects the inspected record. The corrected live audit required exact table/inspector scores and passed all 60 links without relaxing that check.

After that fix, the affected inspector and overlap regression selection passed **12 tests in 41.60 seconds** (`reports/dashboard-inspector-regression-tests.xml`).

All four dashboard scripts pass Node syntax checks and the changes pass `git diff --check`. Browser automation still fails at startup with Windows error 1920. The new importance form, sort selector, trial dialogs, overlap matrix/table, score buttons, pagination, and browser downloads therefore have **not** received a browser click-through. The executed evidence covers real HTTP/data/model behavior; rendered interaction and layout remain unverified.
