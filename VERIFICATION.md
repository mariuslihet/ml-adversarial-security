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
