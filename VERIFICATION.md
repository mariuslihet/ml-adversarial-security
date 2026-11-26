# Executed verification

## Three model tools — 26 November 2026

The complete suite passed **338 tests in 405.16 seconds** (`reports/dashboard-all-lab-tests.xml`). The 27 new tool tests passed independently in 23.76 seconds (`reports/dashboard-lab-tests.xml`). After making stress-test recipes identify the exact selected trial, six affected actual-model/recipe checks passed in 27.08 seconds (`reports/dashboard-lab-recipe-tests.xml`). Tests use authenticated source records, actual fitted model checkpoints, numerical weight/correction fitting, separate validation roles, held-out scores, complete CSVs, and real recipe exports. They independently check the ensemble optimum conditions, saved-attack weighted predictions, calibration coefficients/knots and confidence bands, exact training-derived/zero replacements, new/fixed mistakes, and invalid/stale requests.

The first focused run caught exact zero/one scores from a fitted monotone calibration curve producing a non-finite log loss when clipping retained float32 precision. Metrics now convert probabilities to float64 before clipping, and tests independently calculate finite log loss on those actual fitted outputs. Confidence bands also use explicit float64 arithmetic so browser band selection and server band counts share the same boundaries.

The live audit checked **24 configurations** across the full-data run and a real 128-record experiment: fitted/equal ensembles with saved-attack replay, both correction methods on tree/rounded models, and both missing-value rules on original/tree/rounded models. It independently matched **36,482 predictions**, checked separate fitting/threshold/test source IDs, validated ensemble optimum conditions and real calibration knots, and checked **10,896 CSV rows** plus recipes. Evidence is `reports/dashboard_model_lab_checks.json`; repeat using `scripts/audit_dashboard_lab.py`.

Browser checks covered all three tool switches; member selection and too-few-member validation; fitted and equal weights; actual saved-attack evaluation; test-record paging/mistake filtering; source inspection and Escape close; both confidence methods; confidence-band drilldown/clear and every record filter; median and raw-zero measurement losses; rounded-input scoring; locked column 55; individual/combined trial selection; replacement-value disclosure; and all eleven export actions. The combined four-field median trial introduced eight mistakes, fixed three, and changed eleven decisions on 841 actual test records. Those are measured outcomes, not defaults.

All **eleven browser-saved files** were compared byte-for-byte with freshly recalculated fingerprint-checked exports: three fitted/test-plan recipes, three complete test-record CSVs, three measured summaries, one saved-attack CSV, and one confidence-band CSV. Copies are under `reports/model-lab-browser/`, with hashes and row counts in the live evidence file. The selected stress recipe and record file identify `single-55`, rather than silently exporting the cumulative trial. Desktop layout was visually reviewed. A narrow viewport had no horizontal page overflow in DOM dimensions; narrow screenshot capture was unavailable and the viewport override was reset. The browser console check returned no warnings or errors. This covers the new tools, not a fresh exhaustive pass across older dashboard pages.

## Saved attack replay and source-matched comparisons — 26 November 2026

The full regression suite passed **307 tests in 279.33 seconds** (`reports/dashboard-all-replay-tests.xml`). The 24 replay checks passed independently in 52.51 seconds (`reports/dashboard-replay-tests.xml`); four additional export-validation checks passed in 14.50 seconds (`reports/dashboard-replay-export-tests.xml`). Tests independently restore real model checkpoints for all six saved attacks, including rounded, changed-data, and cleaned models. They check eligibility, escape rates, newly caught records, paired outcomes, all six filters, pagination, complete exports, cross-run intersections and missing rows, reconstructed measurement values, invalid settings, and stale fingerprints. There are no mocked models, attack outcomes, or source records.

The independent live audit passed **348 model/attack checks**: six attacks against all 29 trained models in each of two full-data runs. It checked every results page, filter/record/measurement export responses, **684 reconstructed measurements**, and the cross-run source intersection. The first audit caught a small arithmetic mismatch because the independent process used different CPU thread settings. Matching the verified experiments' deterministic seed/thread settings resolved it; exact aggregate equality was retained. Repeat using `scripts/audit_dashboard_replay.py`; evidence is `reports/dashboard_replay_checks.json`.

Browser input verified full-run replay, all six outcome views, next/previous pages, comparisons across completed runs, rounded-input scoring, cleanup-model selection, source-detail opening, all 57 measurement rows, Escape-to-close, the filter-table disclosure, and all three export buttons. One real comparison displayed 14 changed decisions, zero first-only catches, 14 second-only catches, ten caught in both, and eight missed in both across 32 actual shared rows. A different smaller run shared zero attacked source IDs; the UI correctly excluded 32 first-only and two second-only records. Model choices narrow to those available in both runs.

The browser saved `attack-replay-records.csv` with 32 actual records. The downloaded bytes exactly matched a fresh fingerprint-checked export, and a copy is retained as `reports/browser-attack-replay-records.csv`. Desktop and narrow-screen screenshots were inspected; the large all-filter table now collapses so paired records remain accessible. Narrow-screen controls fit without horizontal page overflow. The browser console check returned no warnings or errors. This verifies the new workflows; it is not a fresh exhaustive acceptance pass across all older pages, and replaying fixed examples does not establish adaptive attack resistance.

## Configuration reuse and selective retries — 26 November 2026

The full suite passed **283 tests in 247.56 seconds**, recorded in `reports/dashboard-reuse-tests.xml`. The 13 added checks use real SQLite history and real full-pipeline subprocesses. They cover persistence, stale-revision rejection, historical settings, fixed-context mismatch refusal (including archive without rebinding), protected writes, verified report capture/export, exact reruns, and selective retries. An actual dataset-limit failure is corrected in a separate retry; another test preserves a completed report while retrying only a cancelled second process. Original entries and report bytes are checked unchanged. No model, source record, or process outcome is mocked.

Browser access recovered during this upgrade. Actual browser input saved/applied a preset, restored a changed form field, created a second configuration revision, opened the first revision, archived/restored the latest preset, and started full runs from a preset, a previous process, and a verified report. It also loaded original settings for correction, captured a verified report as a preset, triggered preset export, and checked the include-archived selector. A browser-created campaign genuinely failed because its requested training changes exceeded available spam records; the retry form corrected the share from 0.49 to 0.01 and the separate retry completed. Its verified comparison displayed five utility rows and twelve attack rows; the original failure remained visible.

The read-only live audit `scripts/audit_dashboard_reuse.py` checks retained revisions and retry lineage, verifies report/artifact fingerprints, reloads all five model variants for the preset run and exact process repeat, and requires zero difference from saved scores. It also requires identical test IDs and all five complete score vectors across the two independently executed pipelines. Evidence is `reports/dashboard_reuse_checks.json`. These 128-record, one-round runs verify execution/reproducibility, not model quality or attack convergence.

The browser console check returned no warnings or errors. Preset export was clicked and its HTTP response independently checked; browser download-event capture expired, so a saved browser download file was not verified. A narrow viewport DOM check found no horizontal page overflow; screenshot capture was unavailable, so visual/mobile and keyboard acceptance remain incomplete. Earlier pages have not all received a fresh exhaustive browser pass. Historical browser-startup blockers below describe earlier audits, not the now-working controls.

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
