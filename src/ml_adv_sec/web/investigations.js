"use strict";

const nx={budget:{model:"clean",category:"all",page:1,size:25},budgetSeq:0,budgetResult:null,cases:{run:"",status:"all",search:"",page:1,size:25},caseSeq:0,caseResult:null,drafts:{},editDrafts:{}};
const nxStatuses=[["open","Open"],["reviewed","Reviewed"],["resolved","Resolved"],["archived","Archived"]];
const nxRunName=run=>(fx.available||[]).find(r=>r.id===run)?.name||run;

function budgetPage() {
  const d=nx.budget,models=[...new Set(state.report.pareto.map(r=>r.model))];
  if(!models.includes(d.model))d.model=models[0];
  return `<section class="panel table-panel segment-panel"><div class="panel-head"><div><h2>What changed at each measured attack budget?</h2><p class="panel-description">Recalculate saved sweep examples against their pinned models. Only levels actually run are shown.</p></div><div class="row-actions"><button id="budget-export-summary" class="button" disabled>Export budget results</button><button id="budget-export-records" class="button primary" disabled>Export record outcomes</button></div></div><div class="record-toolbar"><label>Filter<select id="budget-model">${fxOptions(models.map(name=>[name,modelNames[name]||name]),d.model)}</select></label><label>Records to show<select id="budget-category">${fxOptions([["all","All attacked records"],["escaped","Escaped at a measured level"],["resisted","Caught at every measured level"],["recovered","Caught again at a larger level"],["already-missed","Already missed before attacks"]],d.category)}</select></label><label>Rows per page<select id="budget-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],d.size)}</select></label><div class="research-note">First observed escape is only the first successful saved level. It is not an exact minimum change or proof of robustness.</div></div><p id="budget-status" class="view-status" role="status">Recalculating the saved budget experiments…</p><div id="budget-body" aria-live="polite"></div><div class="page-controls"><span id="budget-range"></span><div class="row-actions"><button id="budget-prev" class="button" disabled>Previous</button><button id="budget-next" class="button" disabled>Next</button></div></div></section>`;
}

async function nxLoadBudget() {
  const context=fxContext(),request=++nx.budgetSeq,view={...nx.budget},query={run:context.run,report_sha256:context.hash,...nx.budget};
  nx.budgetResult=null;
  for(const id of ["budget-export-summary","budget-export-records","budget-prev","budget-next"])document.getElementById(id).disabled=true;
  document.querySelector("#budget-body").innerHTML="";document.querySelector("#budget-range").textContent="";document.querySelector("#budget-status").textContent="Verifying saved examples, model versions, and measured outcomes…";
  try {
    const data=await readJSON("/api/budgets?"+fxQuery(query));
    if(request!==nx.budgetSeq||!fxCurrent(context)||state.page!=="budgets")return;
    nx.budget.page=data.page;nx.budgetResult={data,query};
    document.querySelector("#budget-body").innerHTML=`<div class="confusion-grid">${[["Measured levels",data.levels.length],["Caught before attacks",data.counts.eligible],["Escaped at least one level",data.counts.escaped],["Caught again at a larger level",data.counts.recovered]].map(([name,count])=>`<div><span>${name}</span><strong>${num(count)}</strong><small>Actual saved experiments</small></div>`).join("")}</div>${axTable(["Allowed change","Attacked records","Originally caught","Escapes","Escape rate","Spam caught after attack","First escapes at this level","Caught again"],data.levels.map(r=>`<tr><td>${changeLevel(r.epsilon)}</td><td>${num(r.records)}</td><td>${num(r.eligible)}</td><td>${num(r.escaped)}</td><td>${pct(r.escape_rate)}</td><td>${pct(r.spam_caught)}</td><td>${num(r.newly_escaped)}</td><td>${num(r.caught_again)}</td></tr>`).join(""))}<div class="panel-head analysis-subhead"><div><h2>Record-by-record outcomes</h2><p class="panel-description">Open a score to inspect the exact saved input changes at that level. Non-monotonic outcomes are retained.</p></div></div>${data.rows.length?axTable(["Source row","Original score","First observed escape",...data.levels.map(r=>`Change ${changeLevel(r.epsilon)}`)],data.rows.map(r=>`<tr><td>${num(r.source_row)}${r.eligible?"":'<span class="method-sub">Already missed</span>'}</td><td>${pct(r.before_score,3)}</td><td>${r.first_observed_escape==null?(r.eligible?"No measured escape":"Already missed"):changeLevel(r.first_observed_escape)}</td>${r.levels.map(level=>`<td><button class="text-button budget-inspect" data-source-row="${r.source_row}" data-level="${level.index}">${pct(level.score,3)}</button><span class="method-sub">${level.escaped?"Escaped":level.flagged?"Caught":"Already missed"}</span></td>`).join("")}</tr>`).join("")):'<p class="no-jobs">No records match this view.</p>'}<div class="table-foot">Caught-again counts compare adjacent saved levels. No interpolation or enforced monotonic curve is used. A model that misses a record before an attack does not contribute an escape. Failed finite attacks do not establish certified robustness.</div>`;
    document.querySelector("#budget-status").textContent=`Fresh outcomes match the saved report at every level. Threshold: ${data.threshold>1?"Above 100%":pct(data.threshold,4)}. ${num(data.counts["already-missed"])} records were already missed.`;
    document.querySelector("#budget-range").textContent=`${num(data.total)} matching records · page ${num(data.page)} of ${num(data.pages)}`;
    document.querySelector("#budget-prev").disabled=data.page<=1;document.querySelector("#budget-next").disabled=data.page>=data.pages;
    for(const scope of ["summary","records"])document.getElementById("budget-export-"+scope).disabled=false;
    document.querySelectorAll(".budget-inspect").forEach(button=>button.addEventListener("click",()=>nxBudgetRecord(context,view.model,Number(button.dataset.level),Number(button.dataset.sourceRow))));
  }catch(error){if(request===nx.budgetSeq&&fxCurrent(context)&&state.page==="budgets")document.querySelector("#budget-status").textContent=error.message;}
}

async function nxBudgetRecord(context,model,level,sourceRow) {
  const request=++fx.modalSeq;fxOpenDialog("Budget experiment · source row "+num(sourceRow));
  try {
    const data=await readJSON("/api/budget-record?"+fxQuery({run:context.run,report_sha256:context.hash,model,level,source_row:sourceRow}));
    if(request!==fx.modalSeq||!fxCurrent(context)||!document.querySelector("#feature-dialog").open)return;
    document.querySelector("#feature-dialog-body").innerHTML=`<div class="training-counts"><span>${esc(modelNames[model]||model)}</span><span>Allowed change ${changeLevel(data.epsilon)}</span><span>${data.escaped?"Escaped":data.eligible?"Caught after attack":"Already missed before attack"}</span></div><div class="inspection-scores"><div><span>Original spam score</span><strong>${pct(data.before_score,4)}</strong></div><div><span>After attack</span><strong>${pct(data.after_score,4)}</strong></div><div><span>Largest scaled input change</span><strong>${measurement(data.linf)}</strong><small>Total change length: ${measurement(data.l2)}</small></div></div><label class="csv-choice"><input id="budget-show-all" type="checkbox"> Show unchanged measurements</label><div id="budget-features"></div><div class="table-foot">Scores were recalculated from the exact pinned checkpoint. Input differences use the saved attacked input; the rounded-input model additionally applies its saved rounding step. Original-unit values are recovered from the run's scaler.</div>`;
    const draw=all=>{const rows=data.features.filter(r=>all||r.changed).sort((a,b)=>Math.abs(b.scaled_change)-Math.abs(a.scaled_change));document.querySelector("#budget-features").innerHTML=rows.length?axTable(["Measurement","Before","After","Scaled change","Rule"],rows.map(r=>`<tr><td>${esc(featureName(r.name))}</td><td>${measurement(r.before)}</td><td>${measurement(r.after)}</td><td>${rxSigned(r.scaled_change)}</td><td>${r.locked?"Locked":"Can change"}</td></tr>`).join("")):'<p class="no-jobs">The saved attack left this record unchanged.</p>';};
    draw(false);document.querySelector("#budget-show-all").addEventListener("change",event=>draw(event.target.checked));
  }catch(error){if(request===fx.modalSeq&&fxCurrent(context)&&document.querySelector("#feature-dialog").open)document.querySelector("#feature-dialog-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;}
}

function nxCaseFields(draft,prefix) {
  return `<div class="notebook-fields"><label>Title<input id="${prefix}-title" maxlength="120" required value="${esc(draft.title||"")}"></label><label>Notes<textarea id="${prefix}-notes" maxlength="4000" rows="5">${esc(draft.notes||"")}</textarea></label><label>Tags, separated by commas<input id="${prefix}-tags" maxlength="220" value="${esc((draft.tags||[]).join(", "))}"></label><label>Status<select id="${prefix}-status">${fxOptions(nxStatuses,draft.status||"open")}</select></label></div>`;
}
function nxReadFields(prefix) {return {title:document.getElementById(prefix+"-title").value,notes:document.getElementById(prefix+"-notes").value,tags:document.getElementById(prefix+"-tags").value.split(",").map(tag=>tag.trim()).filter(Boolean),status:document.getElementById(prefix+"-status").value};}

function notebookPage() {
  const d=nx.cases,draft=nx.drafts[state.reportHash]||{};
  const create=state.report?`<form id="case-create-form"><div class="record-toolbar"><label>Filter<select id="case-create-model">${axModelOptions(state.report,draft.model||"clean")}</select></label><label>Source row in test set<input id="case-create-source" type="number" min="1" max="4601" step="1" value="${esc(draft.source_row||"")}" required></label><p class="research-note">New cases bind to ${esc(nxRunName(state.run))} and its current report fingerprint. Model scores and source membership are checked before saving.</p></div>${nxCaseFields(draft,"case-create")}<div class="record-actions"><span id="case-create-status" class="view-status" role="status"></span><button class="button primary" type="submit">Save new case</button></div></form>`:'<p class="no-jobs">Choose a verified run above to save a new case. Existing notes remain available below.</p>';
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>Save an investigation with its evidence</h2><p class="panel-description">Your notes are stored on this computer. Each case keeps an exact source row, model, and report version; opening it rechecks the evidence.</p></div></div><details class="notebook-create"><summary>Create a case from the selected run</summary>${create}</details></section><section class="panel table-panel section-spacer"><div class="panel-head"><h2>Saved cases</h2><button id="cases-export" class="button" disabled>Export matching notes</button></div><div class="record-toolbar"><label>Run<select id="cases-run">${fxOptions([["","Every saved case"],...(fx.available||[]).map(r=>[r.id,r.name])],d.run)}</select></label><label>Status<select id="cases-status">${fxOptions([["all","All statuses"],...nxStatuses],d.status)}</select></label><label>Search titles, notes, or tags<input id="cases-search" type="search" maxlength="200" value="${esc(d.search)}"></label><label>Rows per page<select id="cases-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],d.size)}</select></label></div><p id="cases-message" class="view-status" role="status">Reading the local notebook…</p><div id="cases-body"></div><div class="page-controls"><span id="cases-range"></span><div class="row-actions"><button id="cases-prev" class="button" disabled>Previous</button><button id="cases-next" class="button" disabled>Next</button></div></div><div class="table-foot">The list shows your saved notes and result bindings, not a fresh model check. Open a case to validate its evidence and recalculate scores. Archiving preserves the case and its revision history.</div></section>`;
}

async function nxLoadCases() {
  const request=++nx.caseSeq,query={...nx.cases};nx.caseResult=null;
  for(const id of ["cases-export","cases-prev","cases-next"])document.getElementById(id).disabled=true;
  document.querySelector("#cases-message").textContent="Reading saved notes…";document.querySelector("#cases-body").innerHTML="";document.querySelector("#cases-range").textContent="";
  try {
    const data=await readJSON("/api/cases?"+fxQuery(query));
    if(request!==nx.caseSeq||state.page!=="notebook")return;
    nx.cases.page=data.page;nx.caseResult={data,query};
    document.querySelector("#cases-body").innerHTML=data.rows.length?axTable(["Case","Status","Bound source record","Filter identifier","Tags","Revision","Open"],data.rows.map(r=>`<tr><td class="method-name">${esc(r.title)}<span class="method-sub">${esc(nxRunName(r.run))}</span></td><td>${esc(nxStatuses.find(([key])=>key===r.status)?.[1]||r.status)}</td><td>${num(r.source_row)}<span class="method-sub">Report ${esc(r.report_sha256.slice(0,12))}</span></td><td>${esc(modelNames[r.model]||r.model)}</td><td>${esc(r.tags.join(", "))}</td><td>${num(r.revision)}</td><td><button class="text-button case-open" data-id="${esc(r.id)}">Open case</button></td></tr>`).join("")):'<p class="no-jobs">No saved cases match these filters.</p>';
    document.querySelectorAll(".case-open").forEach(button=>button.addEventListener("click",()=>nxOpenCase(button.dataset.id)));
    document.querySelector("#cases-message").textContent="Notes loaded from the local notebook. Evidence is checked when a case is opened.";
    document.querySelector("#cases-range").textContent=`${num(data.total)} matching cases · page ${num(data.page)} of ${num(data.pages)}`;
    document.querySelector("#cases-prev").disabled=data.page<=1;document.querySelector("#cases-next").disabled=data.page>=data.pages;document.querySelector("#cases-export").disabled=false;
  }catch(error){if(request===nx.caseSeq&&state.page==="notebook")document.querySelector("#cases-message").textContent=error.message;}
}

async function nxOpenCase(id,restoreDraft=true) {
  const request=++fx.modalSeq;fxOpenDialog("Case notebook");
  try {
    const data=await readJSON("/api/case?"+fxQuery({id}));
    if(request!==fx.modalSeq||!document.querySelector("#feature-dialog").open)return;
    const record=data.case,prior=nx.editDrafts[id],draft=restoreDraft&&prior?.revision===record.revision?prior:record;
    document.querySelector("#feature-dialog-title").textContent=record.title;
    document.querySelector("#feature-dialog-body").innerHTML=`<div class="training-counts"><span>${esc(nxRunName(record.run))}</span><span>Source row ${num(record.source_row)}</span><span>Revision ${num(record.revision)}</span></div><p class="view-status">${data.evidence?"Evidence verified. Scores below were recalculated from the bound report's pinned models.":`Evidence unavailable: ${esc(data.evidence_error)}. Your notes are preserved; no replacement model results are shown.`}</p>${data.evidence?axTable(["Filter","Fresh score","Threshold","Decision","Result"],data.evidence.record.models.map(r=>`<tr><td>${esc(fxModelName(r))}</td><td>${pct(r.score,4)}</td><td>${r.threshold>1?"Above 100%":pct(r.threshold,4)}</td><td>${axFlag(r.flagged)}</td><td>${r.correct?"Correct":"Wrong"}</td></tr>`).join("")):""}${prior&&(prior.revision!==record.revision||!restoreDraft)?`<details class="analysis-details"><summary>Your unsaved draft from revision ${num(prior.revision)}</summary><pre class="draft-copy">${esc(JSON.stringify(prior,null,2))}</pre></details>`:""}<form id="case-edit-form">${nxCaseFields(draft,"case-edit")}<div class="record-actions"><div class="row-actions"><button class="button" type="button" id="case-reload">Reload saved version</button><button class="button" type="button" id="case-export">Export saved case and checked evidence</button></div><button class="button primary" type="submit">Save changes</button></div><p id="case-edit-status" class="view-status" role="status">The source/model/report binding is fixed. Saving notes creates a new revision.</p></form><details class="analysis-details"><summary>Revision history · ${num(data.history.length)} saved revisions</summary>${data.history.map(r=>`<article class="case-history"><strong>Revision ${num(r.revision)} · ${esc(r.status)}</strong><p>${esc(new Date(r.updated).toLocaleString())}</p><p>${esc(r.title)}</p><pre>${esc(r.notes)}</pre><small>${esc(r.tags.join(", "))}</small></article>`).join("")}</details>`;
    const form=document.querySelector("#case-edit-form");
    form.addEventListener("input",()=>{nx.editDrafts[id]={...nxReadFields("case-edit"),revision:record.revision};});
    document.querySelector("#case-reload").addEventListener("click",()=>{nx.editDrafts[id]={...nxReadFields("case-edit"),revision:record.revision};nxOpenCase(id,false);});
    form.addEventListener("submit",async event=>{event.preventDefault();if(!form.reportValidity())return;const button=form.querySelector('[type="submit"]');button.disabled=true;const values={id,revision:record.revision,...nxReadFields("case-edit")};nx.editDrafts[id]=values;try{const updated=await postJSON("/api/cases/update",values);delete nx.editDrafts[id];if(request===fx.modalSeq&&document.querySelector("#feature-dialog").open){nxOpenCase(updated.id);if(state.page==="notebook")nxLoadCases();}}catch(error){if(request===fx.modalSeq&&document.querySelector("#feature-dialog").open)document.querySelector("#case-edit-status").textContent=error.message;}finally{if(button.isConnected)button.disabled=false;}});
    document.querySelector("#case-export").addEventListener("click",async event=>{const button=event.currentTarget;button.disabled=true;try{await fxDownload("/api/case-export?"+fxQuery({id}),"case-evidence.json");}catch(error){if(request===fx.modalSeq&&document.querySelector("#feature-dialog").open)document.querySelector("#case-edit-status").textContent=error.message;}finally{if(button.isConnected)button.disabled=false;}});
  }catch(error){if(request===fx.modalSeq&&document.querySelector("#feature-dialog").open)document.querySelector("#feature-dialog-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;}
}

function nxCaptureCase(run,sourceRow,model,hash) {
  const request=++fx.modalSeq;fxOpenDialog("Save source record to notebook");
  document.querySelector("#feature-dialog-body").innerHTML=`<div class="training-counts"><span>${esc(nxRunName(run))}</span><span>Source row ${num(sourceRow)}</span><span>Report ${esc(hash.slice(0,12))}</span></div><form id="case-capture-form">${nxCaseFields({},"case-capture")}<div class="record-actions"><button type="button" class="button" id="case-capture-back">Back to source record</button><button class="button primary" type="submit">Save case</button></div><p id="case-capture-status" class="view-status" role="status">The server verifies this source record and recalculates its pinned model scores before saving the binding.</p></form>`;
  document.querySelector("#case-capture-back").addEventListener("click",()=>axRecordDetail(run,sourceRow,model,hash));
  const form=document.querySelector("#case-capture-form");form.addEventListener("submit",async event=>{event.preventDefault();if(!form.reportValidity())return;const button=form.querySelector('[type="submit"]');button.disabled=true;try{const saved=await postJSON("/api/cases/create",{run,source_row:sourceRow,model,report_sha256:hash,...nxReadFields("case-capture")});if(request===fx.modalSeq&&document.querySelector("#feature-dialog").open)nxOpenCase(saved.id);}catch(error){if(request===fx.modalSeq&&document.querySelector("#feature-dialog").open)document.querySelector("#case-capture-status").textContent=error.message;}finally{if(button.isConnected)button.disabled=false;}});
}

function bindInvestigationContent() {
  bindCampaignContent();
  ++nx.budgetSeq;++nx.caseSeq;nx.budgetResult=null;nx.caseResult=null;
  if(state.page==="budgets") {
    for(const key of ["model","category","size"])document.getElementById("budget-"+key).addEventListener("change",event=>{nx.budget[key]=key==="size"?Number(event.target.value):event.target.value;nx.budget.page=1;nxLoadBudget();});
    document.querySelector("#budget-prev").addEventListener("click",()=>{--nx.budget.page;nxLoadBudget();});document.querySelector("#budget-next").addEventListener("click",()=>{++nx.budget.page;nxLoadBudget();});
    for(const scope of ["summary","records"])document.getElementById("budget-export-"+scope).addEventListener("click",event=>nxExport(event,"budget",scope));
    nxLoadBudget();
  }
  if(state.page==="notebook") {
    for(const key of ["run","status","size"])document.getElementById("cases-"+key).addEventListener("change",event=>{nx.cases[key]=key==="size"?Number(event.target.value):event.target.value;nx.cases.page=1;nxLoadCases();});
    let timer;document.querySelector("#cases-search").addEventListener("input",event=>{nx.cases.search=event.target.value;nx.cases.page=1;++nx.caseSeq;nx.caseResult=null;for(const id of ["cases-export","cases-prev","cases-next"])document.getElementById(id).disabled=true;clearTimeout(timer);timer=setTimeout(()=>{if(state.page==="notebook")nxLoadCases();},250);});
    document.querySelector("#cases-prev").addEventListener("click",()=>{--nx.cases.page;nxLoadCases();});document.querySelector("#cases-next").addEventListener("click",()=>{++nx.cases.page;nxLoadCases();});
    document.querySelector("#cases-export").addEventListener("click",event=>nxExport(event,"case"));
    const form=document.querySelector("#case-create-form"),context=fxContext();
    if(form){form.addEventListener("input",()=>{nx.drafts[context.hash]={...nxReadFields("case-create"),model:document.querySelector("#case-create-model").value,source_row:document.querySelector("#case-create-source").value};});form.addEventListener("submit",async event=>{event.preventDefault();if(!form.reportValidity())return;const button=form.querySelector('[type="submit"]');button.disabled=true;const values={run:context.run,report_sha256:context.hash,model:document.querySelector("#case-create-model").value,source_row:Number(document.querySelector("#case-create-source").value),...nxReadFields("case-create")};try{const saved=await postJSON("/api/cases/create",values);delete nx.drafts[context.hash];if(form.isConnected&&fxCurrent(context)){form.reset();for(const name of ["title","notes","tags","source"])document.getElementById("case-create-"+name).value="";document.querySelector("#case-create-status").textContent="Case saved with verified source and model bindings.";nxLoadCases();nxOpenCase(saved.id);}}catch(error){if(form.isConnected&&fxCurrent(context))document.querySelector("#case-create-status").textContent=error.message;}finally{if(button.isConnected)button.disabled=false;}});}
    nxLoadCases();
  }
}

async function nxExport(event,kind,scope) {
  const button=event.currentTarget,result=kind==="budget"?nx.budgetResult:nx.caseResult,sequence=kind==="budget"?nx.budgetSeq:nx.caseSeq;
  if(!result)return;button.disabled=true;
  const current=()=>button.isConnected&&(kind==="budget"?nx.budgetResult===result&&nx.budgetSeq===sequence:nx.caseResult===result&&nx.caseSeq===sequence);
  try{await fxDownload((kind==="budget"?"/api/budgets.csv?":"/api/cases.csv?")+fxQuery({...result.query,...(scope?{scope}:{})}),kind==="budget"?`budgets-${scope}.csv`:"case-notebook.csv");}
  catch(error){if(current()){document.querySelector(kind==="budget"?"#budget-status":"#cases-message").textContent=error.message;if(kind==="budget"){nx.budgetResult=null;for(const s of ["summary","records"])document.getElementById("budget-export-"+s).disabled=true;}else{nx.caseResult=null;button.disabled=true;}}}
  finally{if(current())button.disabled=false;}
}
