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

## Error breakdowns and data differences — 25 November 2026

The full suite passed **195 tests in 113.94 seconds** (`reports/dashboard-all-segment-tests.xml`). The 52 added checks cover all five original/hardened models plus changed-data and cleaned models, source-based error groups, training-only quantile cuts at three requested band counts, all record-result filters, pagination and clamping, complete filtered record exports, and record-inspector score agreement. Distribution checks independently reconstruct empirical CDF gaps and one-dimensional distribution distances for all 57 measurements across test/validation and all/spam/regular label filters. They check actual source means, medians, training scales, out-of-range counts, histogram membership, exact raw CSV values, invalid parameters, stale report rejection, and an actual-source constant-nonzero subset with exactly zero spread.

The live audit passed across all five saved runs. It executed **105 fresh-model error breakdowns** (seven models and three feature/band settings per run), with fresh test predictions matching saved scores exactly; checked every result filter and complete CSV; paged through every test record at 100 rows per page; and opened each run's record inspector. It also verified **30 partition/label comparisons covering 1,710 measurement distributions** against original source IDs, independent CDF calculations, medians, and range checks. Source-record exports were checked field-for-field against all 57 original measurements. Evidence is `reports/dashboard_segment_checks.json`; repeat with `scripts/audit_dashboard_segments.py`.

Histogram boundaries now include separate below/above-training-range groups. The UI labels a constant value explicitly rather than rounding a tiny successor boundary into a misleading zero-width band. Constant training values have no normalized distance; no artificial scale is substituted. These are descriptive corpus-partition comparisons, not evidence of temporal drift, significance, causality, or future model performance.

All five dashboard scripts pass Node syntax checks and `git diff --check` passes. Browser access was retried through the supported browser tool but failed before page observation with Windows error 1920. New selectors, band buttons, record links, scrolling, keyboard behavior, responsive layout, and actual browser downloads remain unverified through browser input. The HTTP/source/model checks are complete and do not substitute for that browser pass.

## Execution capability upgrade: real campaigns, durable cases, and budget evidence

The full suite passed **270 tests in 174.69 seconds** (`reports/dashboard-capability-tests.xml`). The notebook/budget module passed 54 focused checks; the campaign module passed 21 checks, including real subprocess completion, independent aggregate calculations, pause/restart/resume, cancellation ownership, waiting for a manual process, a genuine dataset-constraint failure, plan validation, and protected write requests. After synchronizing persisted job status with in-memory process ownership under the same lock, two final actual-process/recovery regressions passed in 54.55 seconds (`reports/dashboard-campaign-final-regressions.xml`). No model, library call, process outcome, or source dataset is mocked.

The live dashboard executed a real **four-run campaign**: seeds 53 and 89 crossed with attack-change limits 0.03 and 0.1, using authentic 128-record selections, one training round, and the complete training/attack/poison/defense/evaluation pipeline. It paused after the first successful process, survived an actual dashboard-server restart, and resumed only the three queued entries. Four distinct process IDs and isolated run directories were verified, and the first completed process was not repeated. All reports passed the exact fingerprint saved at process exit. The campaign yielded **20 utility rows and 48 attack rows**, with ten utility groups and 24 attack groups. Means and sample spread were checked against the underlying real results; both CSV exports were verified. These small runs verify execution and queue behavior, not converged robustness or benchmark efficacy.

The live workflow also saved one case from an actual campaign record, using its measured score, true source label, and report fingerprint. It verified the source's 57 measurements and pinned-model scores, saved reviewed/archived revisions, rejected an outdated revision, and exported the checked evidence/history. A further real server restart preserved the three case revisions and all four verified campaign reports. The case is retained as an archived measured-record investigation; it is not default seeded application content. Evidence and process identities are in `reports/dashboard_capability_checks.json`; repeat with `scripts/audit_dashboard_capabilities.py begin`, restart the idle server once the queue is paused, then run the script with `finish`.

Notebook tests use real SQLite transactions and two actual concurrent HTTP edits, proving that only one write can advance a shared revision. A real subsequent model evaluation tests stale-report handling: user notes remain readable while model evidence is unavailable, and no newer report is silently substituted. Budget tests independently recalculate every saved sweep example for all four target models, validate every record filter and measured recovery, verify input differences and immutable fields, and inspect the actual saved example at each level. No missing budget levels or minimum-change answers are invented.

All seven browser scripts pass syntax checks and `git diff --check` passes. Browser access was attempted after the upgrade; the control process still failed before page observation with Windows error 1920. Campaign buttons, notebook forms and revision-recovery UX, budget dialogs, keyboard behavior, responsive layout, and browser-triggered downloads therefore remain unverified through actual browser input. Real HTTP/process/source/model checks passed, but they do not constitute a browser acceptance pass.
