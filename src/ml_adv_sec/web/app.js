"use strict";

const $ = (selector) => document.querySelector(selector);
const content = $("#content");
const state = {run: null, report: null, updated: null, page: "overview", target: "clean", kind: "label-flip", glance: "clean", settings: null, jobs: [], inspect: null, showUnchanged: false};
Object.assign(state, {draft: {}, advancedOpen: false, loading: true, openJobs: new Set()});
let reportRequest = 0, inspectionRequest = 0, reportsRequest = 0, pollingJobs = false, submitting = false;
try {const draft = JSON.parse(sessionStorage.getItem("filter-run-draft") || "{}"); if (draft && typeof draft === "object" && !Array.isArray(draft)) state.draft = draft;} catch (_) { /* Keep the in-memory draft if browser storage is unavailable. */ }
const modelNames = {clean: "Original filter", tree: "Tree filter", pgd: "Attack-trained", trades: "Stability-trained", squeezed: "Rounded inputs", "squeezed-bpda": "Rounded inputs"};
const pages = {overview: ["Overview", "Test results"], attacks: ["Attack tests", "Attack tests"], training: ["Training changes", "Training changes"], details: ["Run details", "About this run"]};
pages.lab = ["Model lab", "Model lab"];
pages["attack-analysis"] = ["Attack analysis", "Attack analysis"];
pages.adaptive = ["Adaptive attacks", "Adaptive attack tests"];
pages.runs = ["Run experiments", "Run an experiment"];
pages.records = ["Record explorer", "Record explorer"];
pages.compare = ["Compare runs", "Compare runs"];
pages.decision = ["Decision lab", "Decision lab"];
pages.disagreements = ["Filter disagreements", "Filter disagreements"];
pages.importance = ["Input importance", "Input importance"];
pages.overlap = ["Attack overlap", "Attack overlap"];
pages.slices = ["Error breakdowns", "Error breakdowns"];
pages.shift = ["Data differences", "Training vs held-out data"];
pages.budgets = ["Attack budgets", "Attack budgets"];
pages.notebook = ["Case notebook", "Case notebook"];
pages.campaigns = ["Experiment campaigns", "Experiment campaigns"];
const series = [{id: "clean", color: "#335ce8"}, {id: "pgd", color: "#278775"}, {id: "trades", color: "#bd7545"}, {id: "squeezed-bpda", color: "#8d6bc6"}];
const number = new Intl.NumberFormat("en");
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const pct = (v, digits = 1) => v == null || !Number.isFinite(Number(v)) ? "—" : (Number(v) * 100).toFixed(digits) + "%";
const num = (v) => v == null || !Number.isFinite(Number(v)) ? "—" : number.format(Number(v));
const changeLevel = v => Number(v).toLocaleString("en", {maximumSignificantDigits: 5});
const label = (name) => esc(modelNames[name] || name);
const infoIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.1"/></svg>';

function meter(value, color = "#335ce8") {
  const width = Math.max(0, Math.min(1, Number(value) || 0)) * 60;
  return `<span class="table-meter"><svg class="mini-meter" width="60" height="4" viewBox="0 0 60 4" aria-hidden="true"><rect width="60" height="4" rx="2" fill="#edf0f6"/><rect width="${width}" height="4" rx="2" fill="${color}"/></svg><span>${pct(value)}</span></span>`;
}

function attackName(row) {
  const id = row.attack || "";
  if (id.includes("cleverhans")) return ["One-step attack", "FGSM · CleverHans"];
  if (id.includes("fgsm")) return ["One-step attack", "FGSM · ART"];
  if (id.includes("pgd")) return ["Repeated-step attack", `PGD · ${id.endsWith("linf") ? "per-feature limit" : "total change limit"}`];
  if (id.includes("cw")) return ["Small-change search", "C&W · optimization"];
  if (id.includes("hsj")) return ["Decision-only attack", "HopSkipJump · decisions only"];
  return [id, ""];
}

function attackRows(target = state.target) {
  const order = ["evasion-fgsm-linf", "evasion-pgd-linf", "evasion-pgd-l2", "evasion-cw-l2", "evasion-hsj-linf", "evasion-cleverhans-fgsm-linf"];
  return state.report.transferability.filter(r => r.target === target).sort((a, b) => order.indexOf(a.attack) - order.indexOf(b.attack));
}

function summaries() {
  const r = state.report;
  const baseline = r.clean.find(x => x.model === "clean");
  const attack = r.transferability.find(x => x.target === "clean" && x.attack === "evasion-pgd-linf");
  const flag = baseline.fpr > r.config.fpr;
  const asr = attack?.asr;
  return `<div class="summary-grid">
    <section class="summary"><div class="summary-label">Spam caught</div><div class="summary-value"><span class="summary-number">${pct(1 - baseline.fnr).replace("%", "")}<span class="summary-unit">%</span></span><span class="tag neutral">Before attacks</span></div><div class="summary-note">Original filter · all test emails</div></section>
    <section class="summary"><div class="summary-label">Regular email flagged</div><div class="summary-value"><span class="summary-number">${pct(baseline.fpr).replace("%", "")}<span class="summary-unit">%</span></span><span class="tag ${flag ? "warning" : ""}">${flag ? "Above" : "Within"} target</span></div><div class="summary-note">Target: ${pct(r.config.fpr)} · checked on unseen emails</div></section>
    <section class="summary"><div class="summary-label">Slipped past the filter</div><div class="summary-value"><span class="summary-number">${asr == null ? "—" : pct(asr).replace("%", "")}<span class="summary-unit">${asr == null ? "" : "%"}</span></span><span class="tag neutral">Under attack</span></div><div class="summary-note">${asr == null ? "No initially caught spam in this attack set" : `${num(Math.round(asr * attack.eligible))} of ${num(attack.eligible)} initially caught spam emails · repeated-step attack`}</div></section>
  </div>`;
}

function chart() {
  const rows = state.report.pareto;
  const budgets = [...new Set(rows.map(r => Number(r.epsilon)))].sort((a, b) => a - b);
  const w = 720, h = 280, left = 54, right = 24, top = 24, bottom = 46;
  const pw = w - left - right, ph = h - top - bottom;
  const minimum = budgets[0], maximum = budgets[budgets.length - 1];
  const x = v => maximum === minimum ? left + pw / 2 : left + ((Number(v) - minimum) / (maximum - minimum)) * pw;
  const y = v => top + (1 - Number(v)) * ph;
  let svg = `<svg class="line-chart" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="chart-title chart-desc"><title id="chart-title">Spam caught as more changes are allowed</title><desc id="chart-desc">Comparison of four filters under attack. Open Attack tests for the full chart values.</desc>`;
  for (let t = 0; t <= 1.001; t += 0.25) {
    svg += `<line class="gridline" x1="${left}" x2="${w - right}" y1="${y(t)}" y2="${y(t)}"/><text x="${left - 12}" y="${y(t) + 4}" text-anchor="end">${Math.round(t * 100)}%</text>`;
  }
  let previousTick = -Infinity;
  for (const b of budgets) if (x(b) - previousTick >= 50 && (b === maximum || x(maximum) - x(b) >= 50)) {svg += `<text x="${x(b)}" y="${h - 18}" text-anchor="middle">${changeLevel(b)}</text>`; previousTick = x(b);}
  svg += `<line class="axis-line" x1="${left}" x2="${w - right}" y1="${y(0)}" y2="${y(0)}"/>`;
  for (const s of series) {
    const points = rows.filter(r => r.model === s.id).sort((a, b) => a.epsilon - b.epsilon);
    const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.epsilon)},${y(p.adversarial_recall)}`).join(" ");
    svg += `<path class="series-line" stroke="${s.color}" d="${d}"/>`;
    for (const p of points) {
      const text = `${modelNames[s.id]} · change ${changeLevel(p.epsilon)} · ${pct(p.adversarial_recall)} spam caught`;
      svg += `<circle cx="${x(p.epsilon)}" cy="${y(p.adversarial_recall)}" r="3.5" stroke="${s.color}" tabindex="0" aria-label="${esc(text)}" data-tip="${esc(text)}"/>`;
    }
  }
  svg += "</svg>";
  return `<section class="panel"><div class="panel-head"><div><h2>Detection under attack</h2><p class="panel-description">Spam caught as the attacker is allowed to change more.</p></div><span class="tag neutral">Higher is better</span></div><div class="chart-key">${series.map(s => `<span class="legend-item"><svg width="14" height="4" viewBox="0 0 14 4" aria-hidden="true"><path d="M0 2h14" stroke="${s.color}" stroke-width="3"/></svg>${label(s.id)}</span>`).join("")}</div><div class="chart-wrap">${svg}</div><div class="chart-caption"><span>Spam caught <strong>(%)</strong></span><span>Change allowed <strong>(scaled units)</strong></span></div></section>`;
}

function glanceBody() {
  const r = state.report.clean.find(r => r.model === state.glance) || state.report.clean[0];
  return `<dl><div class="glance-row"><dt>Spam missed</dt><dd>${pct(r.fnr)}</dd></div><div class="glance-row"><dt>False alarms</dt><dd>${pct(r.fpr)}</dd></div><div class="glance-row"><dt>Accuracy</dt><dd>${pct(r.accuracy)}</dd></div><div class="glance-row"><dt>Processing time</dt><dd>${Number(r.p95_ms).toFixed(2)} <span class="method-sub-inline">ms</span></dd></div></dl><svg class="small-track" width="100%" height="5" viewBox="0 0 240 5" preserveAspectRatio="none" aria-label="${pct(r.accuracy)} accuracy"><rect width="240" height="5" rx="2.5" fill="#edf0f6"/><rect width="${240 * r.accuracy}" height="5" rx="2.5" fill="#335ce8"/></svg><p class="glance-foot">${num(r.n)} test emails. Processing time covers 95% of measured requests.</p>`;
}

function glance() {
  return `<aside class="panel glance"><h2>Filter at a glance</h2><p class="panel-description">Without an attack</p><label class="sr-only" for="glance-model">Filter to inspect</label><select id="glance-model">${state.report.clean.map(r => `<option value="${esc(r.model)}" ${r.model === state.glance ? "selected" : ""}>${label(r.model)}</option>`).join("")}</select><div id="glance-body">${glanceBody()}</div></aside>`;
}

function attacksTable(compact = false) {
  const target = compact ? "clean" : state.target;
  const all = attackRows(target);
  const rows = compact ? all.filter(r => !r.attack.includes("cleverhans") && !r.attack.endsWith("pgd-l2")) : all;
  return `<section class="panel table-panel"><div class="panel-head"><div><div class="subhead"><h2>${compact ? "Attack breakdown" : "What got through"}</h2><span class="count">${rows.length} tests</span></div><p class="panel-description">Same spam emails. Different ways to change them.</p></div>${compact ? '<a class="text-link" href="#attacks">See all tests</a>' : `<div class="row-actions"><label class="sr-only" for="target-filter">Filter to test</label><select id="target-filter"><option value="clean" ${state.target === "clean" ? "selected" : ""}>Original filter</option><option value="tree" ${state.target === "tree" ? "selected" : ""}>Tree filter</option></select></div>`}</div><div class="table-wrap"><table><thead><tr><th>Attack</th><th>Spam tested</th><th>Initially caught</th><th>Slipped through</th><th>Still caught</th>${compact ? "" : "<th>Valid changes</th>"}<th><span class="sr-only">Details</span></th></tr></thead><tbody>${rows.map(r => {const [name, sub] = attackName(r); return `<tr><td><span class="method-name">${esc(name)}</span><span class="method-sub">${esc(sub)}</span></td><td class="tabular">${num(r.attack_n)}</td><td class="tabular">${num(r.eligible)}</td><td>${meter(r.asr)}</td><td class="tabular">${pct(r.adversarial_recall)}</td>${compact ? "" : `<td class="tabular">${pct(r.projected_valid_fraction, 0)}</td>`}<td><button class="text-button inspect-attack" data-attack="${esc(r.attack)}" data-target="${esc(target)}" aria-label="Inspect ${esc(name + ' ' + sub)}">Inspect</button></td></tr>`;}).join("")}</tbody></table></div><div class="table-foot">“Slipped through” counts only spam the filter caught before the attack. A dash means there were no eligible emails.</div></section>`;
}

function sweepTable() {
  return `<section class="panel table-panel section-spacer"><div class="panel-head"><div><h2>The numbers behind the chart</h2><p class="panel-description">Each filter is attacked separately. Rounded inputs are tested with an attack that accounts for rounding.</p></div></div><div class="table-wrap"><table><thead><tr><th>Filter</th><th>Change allowed</th><th>Spam caught</th><th>Slipped through</th><th>F1 score</th><th>Time · 95th percentile</th></tr></thead><tbody>${state.report.pareto.map(r => `<tr><td class="method-name">${label(r.model)}</td><td class="tabular">${changeLevel(r.epsilon)}</td><td>${meter(r.adversarial_recall)}</td><td>${pct(r.asr)}</td><td>${Number(r.clean_f1).toFixed(3)}</td><td>${Number(r.p95_ms).toFixed(2)} ms</td></tr>`).join("")}</tbody></table></div></section>`;
}

function overview() {
  return summaries() + `<div class="chart-grid">${chart()}${glance()}</div>` + attacksTable(true) + `<div class="note-box">${infoIcon}<span>The false-alarm target is set using a separate validation set. Results here come from unseen test emails, so the actual rate can be higher.</span></div>`;
}

function attackPage() {
  return attacksTable() + `<div class="chart-grid section-spacer">${chart()}${glance()}</div>` + sweepTable();
}

function trainingPage() {
  const kinds = {"label-flip": "Changed labels", collision: "Shifted features", backdoor: "Added pattern"};
  const descriptions = {"label-flip": "Some training spam was marked as regular email. The email measurements were left alone.", collision: "Some training spam was changed to look closer to regular email. Its spam label stayed the same.", backdoor: "A small repeated pattern was added to some training spam, which was then marked as regular email."};
  const rows = state.report.poisoning.filter(r => r.kind === state.kind);
  const baseline = state.report.clean.find(r => r.model === "clean");
  const detector = state.report.detector;
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>When the training data changes</h2><p class="panel-description">Compare the affected filter with the same filter trained after suspicious records were removed.</p></div><div class="row-actions"><label class="sr-only" for="change-kind">Training change</label><select id="change-kind">${Object.entries(kinds).map(([id, name]) => `<option value="${id}" ${state.kind === id ? "selected" : ""}>${name}</option>`).join("")}</select></div></div><div class="table-wrap"><table><thead><tr><th>Training set</th><th>Selected</th><th>Spam caught</th><th>Regular email flagged</th><th>F1 score</th><th>Records</th>${state.kind === "backdoor" ? "<th>Pattern slips through</th>" : ""}</tr></thead><tbody><tr><td><span class="method-name">Original training data</span><span class="method-sub">Reference result</span></td><td>—</td><td>${pct(1 - baseline.fnr)}</td><td>${pct(baseline.fpr)}</td><td>${Number(baseline.f1).toFixed(3)}</td><td>—</td>${state.kind === "backdoor" ? "<td>—</td>" : ""}</tr>${rows.map(r => `<tr><td><span class="method-name">${r.sanitized ? "After cleanup" : "Changed training data"}</span><span class="method-sub">${r.sanitized ? `${num(r.filter?.quarantined)} records removed · ${num(r.filter?.poison_detected)} selected records found` : `${kinds[state.kind]} · ${num(r.selected_count)} selected · ${num(r.modified_count)} changed · requested ${pct(r.ratio, 3)}`}</span></td><td>${pct(r.actual_ratio)}</td><td>${meter(1 - r.fnr)}</td><td>${pct(r.fpr)}</td><td>${Number(r.f1).toFixed(3)}</td><td><button class="text-button inspect-training" data-model="${esc(r.model)}">Inspect records</button></td>${state.kind === "backdoor" ? `<td>${pct(r.trigger_asr)}</td>` : ""}</tr>`).join("")}</tbody></table></div><div class="table-foot">${esc(descriptions[state.kind])} Cleanup can also remove unchanged records; it is not assumed to be correct.</div></section>
    <div class="details-grid section-spacer"><section class="panel"><div class="panel-head"><div><h2>Extra input checks</h2><p class="panel-description">Flag unusual inputs and treat them as spam.</p></div></div><dl class="detail-list"><div class="detail-row"><dt>Emails flagged</dt><dd>${num(detector.flagged)} <span class="method-sub">${pct(detector.flag_rate)} of test emails</span></dd></div><div class="detail-row"><dt>Spam caught with these checks</dt><dd>${pct(1 - detector.fnr)}</dd></div><div class="detail-row"><dt>Regular email flagged</dt><dd>${pct(detector.fpr)}</dd></div></dl></section><section class="panel"><div class="panel-head"><h2>How to read this</h2></div><div class="plain-list"><p>“Selected” is the measured share of the original training set chosen for changes. The record counts also show how many actually changed.</p><p>Every filter is checked on the same untouched test emails. F1 combines how much spam was caught with how often the filter was right when it flagged an email.</p><p>Removing records or adding extra checks can make results worse. These tables show what actually happened.</p></div></section></div>`;
}

function detailsPage() {
  const r = state.report, cfg = r.config, data = r.dataset;
  const fileHashes = Object.entries(data.files || {});
  const rawHash = fileHashes.find(([name]) => name.endsWith("spambase.data"))?.[1];
  const detail = (name, value) => `<div class="detail-row"><dt>${esc(name)}</dt><dd>${esc(value)}</dd></div>`;
  return `<div class="details-grid"><section class="panel"><div class="panel-head"><h2>The email records</h2></div><dl class="detail-list">${detail("Source", "UCI Spambase")}${detail("Source records", num(data.raw_rows))}${detail("Unique records available", num(data.deduplicated_rows))}${detail("Records used in this run", num(r.counts.selected_records))}${detail("Training set", num(data.split_sizes.train))}${detail("Validation set", num(data.split_sizes.val))}${detail("Test set", num(data.split_sizes.test))}${detail("Measurements per email", r.counts.features)}</dl></section><section class="panel"><div class="panel-head"><h2>Run settings</h2></div><dl class="detail-list">${detail("Training rounds", cfg.epochs)}${detail("Spam emails attacked", r.counts.attack_samples)}${detail("Spam emails requested", cfg.attack_samples)}${detail("Attack steps", cfg.steps)}${detail("Change limit", cfg.epsilon)}${detail("False-alarm target", pct(cfg.fpr))}${detail("Repeatable starting seed", cfg.seed)}${detail("Processing", "Local CPU")}</dl></section><section class="panel"><div class="panel-head"><h2>What this run tells you</h2></div><div class="plain-list"><p>These are saved measurements from ${num(data.split_sizes.test)} test emails. Attack results use a smaller set of ${num(r.counts.attack_samples)} spam emails.</p><p>The corpus is historical. These results do not establish how well the filter will handle today's mail.</p><p>The attacks change email measurements. A valid change in a table does not prove that a matching email can be written.</p>${r.counts.selected_records < data.deduplicated_rows ? "<p>This run uses a selected part of the available dataset.</p>" : ""}</div></section><section class="panel"><div class="panel-head"><h2>Run record</h2></div><div class="plain-list"><p>${esc(r.platform)}</p><p>Saved ${esc(new Date(state.updated).toLocaleString())}</p><p class="panel-description">Dataset file fingerprint</p><div class="hash">${esc(rawHash || "Not recorded")}</div><p class="panel-description">Export results to keep the full settings and measurements.</p></div></section></div><section class="panel table-panel section-spacer"><div class="panel-head"><h2>Installed tools</h2></div><div class="table-wrap"><table><thead><tr><th>Tool</th><th>Version</th></tr></thead><tbody>${Object.entries(r.packages || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="tabular">${esc(v)}</td></tr>`).join("")}</tbody></table></div></section>`;
}

function bindContent() {
  bindAdaptiveContent();
  bindAttackAnalysis();
  bindModelLab();
  bindAttackReplay();
  $("#glance-model")?.addEventListener("change", e => {state.glance = e.target.value; $("#glance-body").innerHTML = glanceBody();});
  $("#target-filter")?.addEventListener("change", e => {state.target = e.target.value; render();});
  $("#change-kind")?.addEventListener("change", e => {state.kind = e.target.value; render();});
  document.querySelectorAll(".inspect-attack").forEach(button => button.addEventListener("click", () => inspectAttack(button.dataset.attack, 0, button.dataset.target)));
  $("#run-form")?.addEventListener("submit", launchRun);
  const form = $("#run-form");
  if (form) {
    for (const [name, value] of Object.entries(state.draft)) {const field = form.elements.namedItem(name); if (field && "value" in field) field.value = value;}
    form.addEventListener("input", () => {state.draft = Object.fromEntries(new FormData(form).entries()); try {sessionStorage.setItem("filter-run-draft", JSON.stringify(state.draft));} catch (_) { /* Draft remains available in this tab. */ }});
    const advanced = form.querySelector("details"); advanced.open = state.advancedOpen;
    advanced.addEventListener("toggle", () => state.advancedOpen = advanced.open);
  }
  bindJobActions();
  bindFeatureContent();
  const tip = $("#tooltip");
  for (const point of document.querySelectorAll("[data-tip]")) {
    const show = () => {const rect = point.getBoundingClientRect(); tip.textContent = point.dataset.tip; tip.hidden = false; tip.style.left = Math.max(10, Math.min(rect.x - 80, window.innerWidth - 250)) + "px"; tip.style.top = Math.max(10, rect.y - 52) + "px";};
    point.addEventListener("mouseenter", show);
    point.addEventListener("focus", show);
    point.addEventListener("mouseleave", () => tip.hidden = true);
    point.addEventListener("blur", () => tip.hidden = true);
  }
}

function render() {
  $("#tooltip").hidden = true;
  const page = pages[state.page] || pages.overview;
  document.title = `Filter Lab — ${page[0]}`;
  $("#breadcrumb-page").textContent = page[0];
  $("#page-title").textContent = page[1];
  document.querySelectorAll("nav a").forEach(a => {a.classList.toggle("active", a.dataset.page === state.page); if (a.dataset.page === state.page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");});
  if (!state.report && !["runs", "notebook", "campaigns"].includes(state.page)) {
    $("#subtitle").textContent = state.loading ? "Checking the saved results and their evidence…" : "Choose a verified saved run or start an experiment.";
    content.innerHTML = state.loading ? '<section class="empty"><p>Checking the saved data, models, and results…</p></section>' : '<section class="empty"><h2>No verified results selected</h2><p>Choose a saved run above, or start a new experiment.</p><a class="button primary" href="#runs">Run an experiment</a></section>';
    content.setAttribute("aria-busy", String(state.loading)); return;
  }
  $("#subtitle").textContent = state.page === "campaigns" ? "Queue actual local experiments and compare their verified results." : state.page === "notebook" ? "Local notes bound to actual records and saved model results." : state.page === "runs" ? "Train the filters, run the attacks, and measure the results on this computer." : `Spambase · ${num(state.report.dataset.split_sizes.test)} test emails · ${num(state.report.counts.attack_samples)} spam emails attacked`;
  content.innerHTML = ({overview, attacks: attackPage, training: trainingPage, details: detailsPage, runs: runPage, records: recordsPage, compare: comparisonPage, decision: decisionLabPage, disagreements: disagreementPage, importance: importancePage, overlap: overlapPage, slices: slicesPage, shift: shiftPage, budgets: budgetPage, notebook: notebookPage, campaigns: campaignPage, lab: modelLabPage, adaptive: adaptivePage, "attack-analysis": attackAnalysisPage}[state.page] || overview)();
  content.setAttribute("aria-busy", "false");
  $("#updated").textContent = state.updated ? `Results saved ${new Date(state.updated).toLocaleString(undefined, {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"})}` : "";
  bindContent();
}

async function readJSON(url) {
  const response = await fetch(url, {cache: "no-store"});
  const result = await response.json().catch(() => ({error: "The server returned an unreadable response."}));
  if (!response.ok) throw new Error(result.error || "The saved results could not be read.");
  return result;
}

function clearReport(loading = false) {
  state.report = null; state.reportHash = null; state.updated = null; state.loading = loading;
  $("#download").disabled = true;
  $("#updated").textContent = "";
  inspectionRequest++;
  if ($("#inspector").open) $("#inspector").close();
  closeFeatureDialog();
  render();
}

async function loadRun(id) {
  const request = ++reportRequest;
  state.run = id;
  clearReport(true);
  try {
    const payload = await readJSON(`/api/report/${encodeURIComponent(id)}`);
    if (request !== reportRequest) return false;
    const r = payload.report;
    if (!r.counts || !payload.evidence?.report_sha256 || !Array.isArray(r.clean) || !r.clean.find(x => x.model === "clean") || !Array.isArray(r.pareto) || !Array.isArray(r.transferability) || !Array.isArray(r.poisoning) || !r.dataset?.split_sizes || !r.config) throw new Error("This run is missing verified results. Complete the evaluation and reload.");
    state.report = r; state.reportHash = payload.evidence.report_sha256; state.updated = payload.updated; state.loading = false;
    $("#run-select").value = id;
    $("#message").hidden = true;
    $("#download").disabled = false;
    render();
    return true;
  } catch (e) {
    if (request !== reportRequest) return false;
    clearReport(false);
    showError(e.message);
    return false;
  }
}

function showError(message) {$("#message").textContent = message; $("#message").hidden = false;}

async function reload(preferred = null) {
  const request = ++reportsRequest;
  $("#reload").disabled = true;
  try {
    const runs = await readJSON("/api/reports");
    if (request !== reportsRequest) return false;
    fx.available = runs;
    const select = $("#run-select");
    select.innerHTML = runs.map(r => `<option value="${esc(r.id)}">${esc(r.name)}</option>`).join("");
    select.disabled = !runs.length;
    if (!runs.length) {++reportRequest; state.run = null; clearReport(false); return false;}
    const id = runs.find(r => r.id === preferred)?.id || runs.find(r => r.id === state.run)?.id || runs.find(r => !r.error)?.id || runs[0].id;
    select.value = id;
    return await loadRun(id);
  } catch (e) {if (request === reportsRequest) {++reportRequest; clearReport(false); showError(e.message);} return false;}
  finally {if (request === reportsRequest) $("#reload").disabled = false;}
}

$("#run-select").addEventListener("change", e => {++reportsRequest; $("#reload").disabled = false; loadRun(e.target.value);});
$("#reload").addEventListener("click", () => reload());
$("#download").addEventListener("click", async () => {
  if (!state.report) return;
  const selected = state.run;
  $("#download").disabled = true;
  try {
    // Recheck disk evidence at export time, then export exactly what is displayed.
    if (!await loadRun(selected) || state.run !== selected) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(state.report, null, 2)], {type: "application/json"}));
    const a = document.createElement("a"); a.href = url; a.download = "filter-results.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {clearReport(false); showError(e.message);}
  finally {$("#download").disabled = !state.report;}
});
window.addEventListener("hashchange", () => {const hash = location.hash.slice(1); if (hash === "main") return; closeFeatureDialog(); state.page = Object.hasOwn(pages, hash) ? hash : "overview"; render(); window.scrollTo(0, 0);});
window.addEventListener("keydown", e => {if (e.key === "Escape") $("#tooltip").hidden = true;});
state.page = Object.hasOwn(pages, location.hash.slice(1)) ? location.hash.slice(1) : "overview";
render();
reload();

function runPage() {
  const cfg = state.settings;
  if (!cfg) return '<section class="empty"><p>Loading run settings…</p></section>';
  const field = (key, name, min, max, step = 1, help = "") => `<label class="form-field"><span>${name}</span><input name="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${esc(cfg[key])}" required>${help ? `<small>${help}</small>` : ""}</label>`;
  return `<section class="panel"><div class="panel-head"><div><h2>New experiment</h2><p class="panel-description">Starts a full local run. Existing results are kept.</p></div><span class="tag neutral">CPU</span></div><form id="run-form"><div class="form-grid"><label class="form-field wide"><span>Run name</span><input name="name" maxlength="80" value="Local experiment" required></label>${field("epochs", "Training rounds", 1, 200)}${field("max_rows", "Records to use", 0, 4601, 1, "0 uses all available records. Smaller runs need at least 128.")}${field("attack_samples", "Spam emails to attack", 1, 512)}${field("steps", "Attack steps", 1, 300)}${field("epsilon", "Change allowed", 0.0001, 2, 0.0001, "Measured after scaling the email measurements.")}<label class="form-field"><span>False-alarm target (%)</span><input name="fpr_percent" type="number" min="0" max="25" step="0.01" value="${cfg.fpr * 100}" required><small>Chosen on validation emails, then checked on the test set.</small></label></div><details class="advanced"><summary>More settings</summary><div class="form-grid">${field("learning_rate", "Learning rate", 0.000001, 0.1, 0.000001)}${field("batch_size", "Emails per training batch", 8, 2048)}${field("hidden", "Model width", 8, 512)}${field("seed", "Repeatable starting seed", 0, 2147483647)}${field("restarts", "Attack restarts", 0, 10)}${field("cw_steps", "Small-change search steps", 1, 500)}${field("hsj_steps", "Decision-only search steps", 1, 100)}${field("collision_steps", "Training change search steps", 1, 300)}${field("quantum", "Rounding step", 0.0001, 2, 0.0001)}<label class="form-field"><span>Change levels to compare</span><input name="epsilons" value="${esc(cfg.epsilons.join(", "))}" required><small>Separate values with commas.</small></label><label class="form-field"><span>Share of training records to change</span><input name="poison_ratios" value="${esc(cfg.poison_ratios.join(", "))}" required><small>Fractions: 0.01 means 1%.</small></label></div></details><div class="form-bottom"><p>Uses the downloaded dataset. Training and attacks run here; no outside service is called.</p><button type="submit" class="button primary" id="start-run" ${(submitting || state.jobs.some(j => j.status === "running")) ? "disabled" : ""}>Start run</button></div><p id="run-error" class="form-error" role="alert" hidden></p></form></section><section class="panel section-spacer"><div class="panel-head"><div><h2>Run history</h2><p class="panel-description">Actual process status and output, including failed or stopped runs.</p></div></div><div id="jobs-list">${jobList()}</div></section>`;
}

const stages = {prepare: "Preparing the dataset", train: "Training the filters", "attack-evasion": "Running attack tests", "attack-poison": "Testing changed training data", defend: "Training and testing defenses", evaluate: "Measuring and saving results"};

function jobList(jobs = state.jobs) {
  if (!jobs.length) return '<p class="no-jobs">No experiments have been started from this dashboard.</p>';
  return jobs.map(j => {const elapsed = Math.max(0, Math.round((new Date(j.finished || Date.now()) - new Date(j.started)) / 1000)); return `<article class="job-row" data-job-id="${esc(j.id)}"><div class="job-heading"><div><h3>${esc(j.name)}</h3><p>${esc(j.status === "running" ? stages[j.stage] || "Starting the process" : ({complete: "Finished", failed: "Failed", interrupted: "Tracking interrupted", cancelled: "Stopped"}[j.status] || j.status))} · <span data-elapsed>${num(elapsed)}</span> seconds · started ${esc(new Date(j.started).toLocaleString())}</p></div><div class="row-actions"><span class="tag ${j.status === "complete" ? "" : "neutral"}">${esc(j.status)}</span>${j.status === "running" ? `<button class="button stop-job" data-id="${esc(j.id)}">Stop run</button>` : j.status === "complete" && j.report_id ? `<button class="button view-job" data-report-id="${esc(j.report_id)}">${j.report_updated ? "Open updated results" : "Open results"}</button>` : ""}</div></div>${j.report_updated ? '<p class="history-note">The report was rechecked after this process finished. The button opens the updated results.</p>' : ""}<details ${state.openJobs.has(j.id) ? "open" : ""}><summary>Process output</summary><pre class="job-log">${esc(j.log || "Waiting for the process to write output.")}</pre></details></article>`;}).join("");
}

async function postJSON(url, values) {
  const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json", "X-Dashboard-Token": $("meta[name=dashboard-token]").content}, body: JSON.stringify(values)});
  const result = await response.json().catch(() => ({error: "The request could not be completed. Reload the dashboard and try again."}));
  if (!response.ok) throw new Error(result.error || "The request failed.");
  return result;
}

async function launchRun(event) {
  event.preventDefault();
  const form = event.currentTarget, button = $("#start-run"), error = $("#run-error");
  if (submitting) return;
  submitting = true; button.disabled = true; error.hidden = true;
  const values = {};
  for (const [key, value] of new FormData(form).entries()) {
    if (key === "name") values.name = value;
    else if (["poison_ratios", "epsilons"].includes(key)) values[key] = value.split(/[,\s]+/).filter(Boolean).map(Number);
    else if (key === "fpr_percent") values.fpr = Number(value) / 100;
    else values[key] = Number(value);
  }
  try {await postJSON("/api/jobs", values); await pollJobs();}
  catch (e) {error.textContent = e.message; error.hidden = false;}
  finally {submitting = false; if (button.isConnected) button.disabled = state.jobs.some(j => j.status === "running");}
}

function bindJobActions() {
  bindRepeatButtons();
  document.querySelectorAll(".stop-job:not([data-bound])").forEach(b => {b.dataset.bound = "true"; b.addEventListener("click", async () => {b.disabled = true; try {await postJSON("/api/jobs/cancel", {id: b.dataset.id});} catch (e) {showError(e.message);} finally {await pollJobs(); if (b.isConnected) b.disabled = false;}});});
  document.querySelectorAll(".view-job:not([data-bound])").forEach(b => {b.dataset.bound = "true"; b.addEventListener("click", async () => {
    b.disabled = true;
    try {++reportRequest; state.run = b.dataset.reportId; clearReport(true); location.hash = "overview"; await reload(b.dataset.reportId);}
    catch (e) {showError(e.message);}
    finally {if (b.isConnected) b.disabled = false;}
  });});
  document.querySelectorAll(".job-row details:not([data-bound])").forEach(details => {details.dataset.bound = "true"; details.addEventListener("toggle", () => {const id = details.closest(".job-row").dataset.jobId; if (details.open) state.openJobs.add(id); else state.openJobs.delete(id);});});
}

async function pollJobs() {
  if (pollingJobs) return;
  pollingJobs = true;
  try {
    state.jobs = await readJSON("/api/jobs");
    const running = state.jobs.find(j => j.status === "running");
    const banner = $("#job-banner");
    banner.hidden = !running;
    if (running) banner.innerHTML = `<span><strong>${esc(running.name)}</strong> · ${esc(stages[running.stage] || "Starting the process")}</span><a href="#runs">View run</a>`;
    const list = $("#jobs-list");
    if (list) {
      if (!state.jobs.length) list.innerHTML = jobList();
      else {
        list.querySelector(".no-jobs")?.remove();
        const rows = new Map([...list.querySelectorAll(".job-row")].map(row => [row.dataset.jobId, row]));
        for (const [index, job] of state.jobs.entries()) {
          const signature = JSON.stringify([job.status, job.stage, job.log, job.finished, job.report_id]);
          let row = rows.get(job.id);
          if (!row || row.dataset.signature !== signature) {
            if (row) {if (row.querySelector("details").open) state.openJobs.add(job.id); else state.openJobs.delete(job.id);}
            const holder = document.createElement("div"); holder.innerHTML = jobList([job]);
            const replacement = holder.firstElementChild;
            replacement.dataset.signature = signature;
            if (state.openJobs.has(job.id)) replacement.querySelector("details").open = true;
            if (row) row.replaceWith(replacement); else list.append(replacement);
            row = replacement;
          }
          if (list.children[index] !== row) list.insertBefore(row, list.children[index] || null);
          row.querySelector("[data-elapsed]").textContent = num(Math.max(0, Math.round((new Date(job.finished || Date.now()) - new Date(job.started)) / 1000)));
        }
        const ids = new Set(state.jobs.map(j => j.id));
        rows.forEach((row, id) => {if (!ids.has(id)) row.remove();});
        bindJobActions();
      }
    }
    if ($("#start-run")) $("#start-run").disabled = submitting || !!running;
  } catch (e) {if (state.page === "runs") showError("Could not check the local process. The page will try again.");}
  finally {pollingJobs = false;}
}

function featureName(name) {
  if (name.startsWith("word_freq_")) return `Word “${name.slice(10)}” frequency`;
  if (name.startsWith("char_freq_")) return `Character “${name.slice(10)}” frequency`;
  return {capital_run_length_average: "Average capital run", capital_run_length_longest: "Longest capital run", capital_run_length_total: "Capital letters total"}[name] || name;
}

const measurement = v => Math.abs(v) < 0.000001 ? "0" : Number(v.toPrecision(6)).toLocaleString("en", {maximumFractionDigits: 6});

function featureTable(data) {
  const rows = data.features.filter(f => state.showUnchanged || f.changed).sort((a, b) => Math.abs(b.scaled_change) - Math.abs(a.scaled_change));
  if (!rows.length) return '<p class="no-jobs">This attack left this email unchanged after the limits were applied.</p>';
  return `<div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Before</th><th>After</th><th>Change</th><th>Rule</th></tr></thead><tbody>${rows.map(f => `<tr><td><span class="method-name">${esc(featureName(f.name))}</span><span class="method-sub">${esc(f.name)}</span></td><td class="tabular">${measurement(f.before)}</td><td class="tabular">${measurement(f.after)}</td><td class="tabular">${f.change > 0.000001 ? "+" : ""}${measurement(f.change)}</td><td>${f.locked ? "Locked" : "Can change"}</td></tr>`).join("")}</tbody></table></div>`;
}

async function inspectAttack(attack, index, target = state.target) {
  const dialog = $("#inspector");
  if (!dialog.open) dialog.showModal();
  const container = $("#inspector-content");
  container.innerHTML = '<p class="no-jobs">Reading the saved attack and checking the model…</p>';
  const run = state.run;
  const reportHash = state.reportHash;
  const request = ++inspectionRequest;
  try {
    const data = await readJSON(`/api/inspect?run=${encodeURIComponent(run)}&attack=${encodeURIComponent(attack)}&sample=${index}&target=${encodeURIComponent(target)}&report_sha256=${encodeURIComponent(reportHash)}`);
    if (request !== inspectionRequest || !dialog.open || state.run !== run) return;
    state.inspect = {attack, index, data};
    const changed = data.features.filter(f => f.changed).length;
    container.innerHTML = `<div class="inspector-controls"><label>Email in attack set <select id="sample-select">${Array.from({length: data.count}, (_, i) => `<option value="${i}" ${i === index ? "selected" : ""}>Email ${i + 1} of ${data.count}</option>`).join("")}</select></label><span>Source record ${num(data.source_row + 1)}</span></div><div class="inspection-scores"><div><span>Before the attack</span><strong>${data.caught_before ? "Caught" : "Missed"}</strong><small title="${esc(data.before)}">Spam score ${pct(data.before)}</small></div><div><span>After the attack</span><strong>${data.caught_after ? "Caught" : "Missed"}</strong><small title="${esc(data.after)}">Spam score ${pct(data.after)}</small></div><div><span>Decision threshold</span><strong title="${esc(data.threshold)}">${data.threshold > 1 ? "Above 100%" : pct(data.threshold)}</strong><small>${label(target)}</small></div></div><div class="feature-heading"><p>${num(changed)} of ${num(data.features.length)} measurements changed.</p><label><input id="show-unchanged" type="checkbox" ${state.showUnchanged ? "checked" : ""}> Show unchanged measurements</label></div><div id="feature-table">${featureTable(data)}</div><div class="table-foot">These are the saved before-and-after numbers, converted back to the dataset's original units. Model scores were calculated again from the saved weights.</div>`;
    $("#sample-select").addEventListener("change", e => inspectAttack(attack, Number(e.target.value), target));
    $("#show-unchanged").addEventListener("change", e => {state.showUnchanged = e.target.checked; $("#feature-table").innerHTML = featureTable(data);});
  } catch (e) {if (request === inspectionRequest && dialog.open) {clearReport(false); showError(e.message);}}
}

$("#inspector").addEventListener("close", () => {if (!$("#inspector").open) inspectionRequest++;});
$("#inspector-close").addEventListener("click", () => $("#inspector").close());
$("#inspector").addEventListener("click", e => {if (e.target === e.currentTarget) {const rect = e.currentTarget.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) e.currentTarget.close();}});
readJSON("/api/settings").then(cfg => {state.settings = cfg; if (["runs","campaigns"].includes(state.page)) render();}).catch(e => showError(e.message));
pollJobs();
setInterval(pollJobs, 3000);
