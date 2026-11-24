# New dashboard workflows: browser verification follow-up

Date: 24 November 2026. Target: http://127.0.0.1:8873.

**Verdict: browser acceptance blocked.** The browser-control process failed before any page observation or input. The supported reset/retry failed with Windows `CreateProcessWithLogonW failed: 1920`; the desktop-control fallback also reported `trusted Node process exited unexpectedly`. The user restarted Codex and reopened the chat. Initialization still failed after that restart and another supported reset. No new browser clicks, screenshots, downloads, keyboard checks, or viewport measurements are claimed.

## Fixes from code tracing

| Finding | Before | Change | Browser verification |
| --- | --- | --- | --- |
| Record export finishing during a new search | The old export could re-enable its button while the new record request was pending. Its error could clear a different selected report. | Capture the run, report hash, view, and request generation. Update status, report state, and button state only if the initiating view is still current. | Pending |
| Comparison export finishing after navigation | An old export failure could replace a newer comparison or try writing into a removed page element. | Guard completion by the initiating run, report hash, request generation, comparison object, and connected button. | Pending |
| Search debounce allowed actions on a changing view | Export, recalculation, and paging remained enabled during the search delay; an older response could restore obsolete results. | Invalidate the previous request and disable dependent actions immediately when the search text changes. The next valid response owns re-enabling them. | Pending |

Both JavaScript files pass Node syntax checks. These are code-trace fixes; syntax checks and server tests do not establish their rendered browser behavior.

## Browser checklist still required

| Area | Controls and workflows | Status |
| --- | --- | --- |
| Record explorer | Every model option, all labels and outcomes, numeric/empty/invalid search and recovery, all sort orders and page sizes, Previous/Next boundaries, Clear filters | Not run |
| Record scores | Recalculate scores for original, tree, hardened, rounded, changed-data, and cleaned models; compare displayed values to pinned-model predictions | Not run |
| Record exports | Export all matching rows rather than just the visible page; with/without original measurements; empty selection; inspect downloaded CSV contents | Not run |
| Run comparisons | Every saved-run selection, same-run zero differences, matching/different test and attacked records, attack limits and eligible counts, comparison CSV download | Not run |
| Training inspector | Every change type and share, before/after cleanup, every list mode, pagination, Inspect record, four actual scores, changed/unchanged measurements toggle, Back to records | Not run |
| Dialog behavior | Close button, Escape, outside click, focus handling, navigation and saved-run changes during loading | Not run |
| Timing and errors | Search during export/recalculation, comparison changes during export, navigation during pending requests, stale report rejection and recovery | Not run |
| Shared controls | Navigation, reload, saved-run selection, report export, actual new experiment completion and opening results | Not run |
| Layout and keyboard | Desktop/tablet/phone layouts, contained table scrolling, visible focus, keyboard access to every new control, page errors | Not run |

## Real-data checks

The repeated analysis-view regression suite trains real models on authentic Spambase records. It checks filtered records, complete CSV contents, original measurements, cohort matching, cleanup masks, and four independently recalculated scores per training detail. Its machine-readable output is `reports/dashboard-browser-followup-tests.xml`.

Result: **44 tests passed in 52.18 seconds**, with no failures or errors.

The live-server audit repeats score recalculation for all five original/hardened models across all five saved runs, training inspection for every changed/cleaned model, and full original-measurement CSV exports. Its output is `reports/dashboard_feature_checks.json`. This is HTTP/model evidence, not a browser pass.

Result: **five saved runs passed**. All **25 score vectors (13,385 predictions)** recalculated from the pinned weights matched saved results exactly. Training details for all **108 changed/cleaned models** and complete CSV exports passed the live checks.
