"use strict";

// Definitions load before app.js; the existing app owns the selected report.
const fx = {records:{model:"clean",label:"all",outcome:"all",search:"",sort:"score-desc",size:25,page:1,features:false},recordSeq:0,compareSeq:0,modalSeq:0,compareWith:null,comparison:null,training:null};
const fxOutcomes = {caught:"Spam caught",missed:"Spam missed","false-alarm":"Regular email flagged",clear:"Regular email left alone"};
const fxKinds = {"label-flip":"Changed labels",collision:"Shifted measurements",backdoor:"Added pattern"};
const fxOptions = (items, selected) => items.map(([value,name])=>`<option value="${esc(value)}" ${String(value)===String(selected)?"selected":""}>${esc(name)}</option>`).join("");
function fxModelName(row) {
  return modelNames[row.model] || `${fxKinds[row.kind]} · requested ${pct(row.ratio,3)}${row.sanitized?" · after cleanup":""}`;
}
function fxModelOptions(selected) {
  return `<optgroup label="Original and hardened filters">${fxOptions(state.report.clean.map(r=>[r.model,fxModelName(r)]),selected)}</optgroup><optgroup label="Changed training data">${fxOptions(state.report.poisoning.map(r=>[r.model,fxModelName(r)]),selected)}</optgroup>`;
}
function fxContext() {return {run:state.run,hash:state.reportHash};}
function fxCurrent(context) {return state.run===context.run && state.reportHash===context.hash;}
function fxQuery(values) {return new URLSearchParams(Object.fromEntries(Object.entries(values).map(([k,v])=>[k,String(v)]))).toString();}
function fxDisableRecordActions() {
  for(const id of ["records-recheck","records-export","records-prev","records-next"]) {
    const button=document.getElementById(id); if(button) button.disabled=true;
  }
}

function recordsPage() {
  if(![...state.report.clean,...state.report.poisoning].some(r=>r.model===fx.records.model)) fx.records.model="clean";
  const r=fx.records;
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>Every test record, with its actual score</h2><p class="panel-description">Review missed spam and false alarms. The source rows come from the original dataset.</p></div></div><div class="record-toolbar"><label>Filter to review<select id="records-model">${fxModelOptions(r.model)}</select></label><label>Actual label<select id="records-label">${fxOptions([["all","All email"],["spam","Spam"],["regular","Regular email"]],r.label)}</select></label><label>Result<select id="records-outcome">${fxOptions([["all","All results"],...Object.entries(fxOutcomes)],r.outcome)}</select></label><label>Find source row<input type="search" inputmode="numeric" id="records-search" value="${esc(r.search)}" placeholder="Source row number"></label></div><div id="record-counts" class="confusion-grid" aria-live="polite"></div><div class="record-actions"><label>Order<select id="records-sort">${fxOptions([["score-desc","Highest spam score"],["score-asc","Lowest spam score"],["source","Source row number"]],r.sort)}</select></label><div class="row-actions"><button class="button" id="records-clear">Clear filters</button><button class="button" id="records-recheck" disabled>Recalculate scores</button><button class="button primary" id="records-export" disabled>Export matching records</button></div></div><label class="csv-choice"><input type="checkbox" id="records-features" ${r.features?"checked":""}> Include original measurements in the CSV</label><p id="records-status" class="view-status" role="status">Reading the verified scores…</p><div id="record-frame"></div><div class="page-controls"><span id="record-range"></span><div class="row-actions"><label>Rows per page<select id="records-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],r.size)}</select></label><button class="button" id="records-prev" disabled>Previous</button><button class="button" id="records-next" disabled>Next</button></div></div><div class="table-foot">The counts cover the whole test set. Filters apply to the record list and CSV export. Original message text is not included in this dataset.</div></section>`;
}

async function fxLoadRecords(recheck=false) {
  const context=fxContext(), request=++fx.recordSeq, view={...fx.records};
  const frame=document.querySelector("#record-frame");
  if(!frame) return;
  fxDisableRecordActions();
  if(view.search.trim() && !/^[0-9]{1,24}$/.test(view.search.trim())) {
    document.querySelector("#records-status").textContent="Enter a source row number using digits, or clear the search.";
    frame.innerHTML='<p class="no-jobs">The source row search accepts numbers only.</p>';
    document.querySelector("#record-range").textContent="";
    frame.setAttribute("aria-busy","false");
    return;
  }
  document.querySelector("#records-status").textContent=recheck?"Recalculating scores from the saved model weights…":"Reading the verified scores…";
  frame.setAttribute("aria-busy","true");
  try {
    const data=await readJSON("/api/records?"+fxQuery({run:context.run,report_sha256:context.hash,...view,recheck:recheck?1:0}));
    if(request!==fx.recordSeq||!fxCurrent(context)||state.page!=="records") return;
    fx.records.page=data.page;
    document.querySelector("#record-counts").innerHTML=Object.entries(fxOutcomes).map(([key,name])=>`<div><span>${esc(name)}</span><strong>${num(data.counts[key])}</strong><small>All ${num(data.tested)} test records</small></div>`).join("");
    frame.innerHTML=data.rows.length?`<div class="table-wrap"><table><thead><tr><th>Source row</th><th>Actual label</th><th>Filter decision</th><th>Spam score</th><th>Result</th><th>Details</th></tr></thead><tbody>${data.rows.map(r=>`<tr><td class="method-name">${num(r.source_row)}</td><td>${r.actual==="spam"?"Spam":"Regular email"}</td><td>${r.decision==="flagged"?"Flagged":"Passed"}</td><td class="tabular" title="${esc(r.score)}">${pct(r.score)}</td><td><span class="tag ${["missed","false-alarm"].includes(r.outcome)?"warning":"neutral"}">${esc(fxOutcomes[r.outcome])}</span></td><td><button class="text-button inspect-source" data-source-row="${r.source_row}">Inspect record</button></td></tr>`).join("")}</tbody></table></div>`:'<p class="no-jobs">No records match these filters.</p>';
    document.querySelector("#record-range").textContent=`${data.total?num((data.page-1)*data.size+1):"0"}–${num(Math.min(data.page*data.size,data.total))} of ${num(data.total)} matching records · ${num(data.tested)} tested`;
    document.querySelector("#records-prev").disabled=data.page<=1;
    document.querySelector("#records-next").disabled=data.page>=data.pages;
    document.querySelector("#records-recheck").disabled=false;
    document.querySelector("#records-export").disabled=false;
    document.querySelector("#records-status").textContent=data.scores_recalculated?`Recalculated all ${num(data.tested)} scores from the pinned model. Largest difference from the saved scores: ${data.maximum_score_difference.toExponential(2)}.`:`Saved model scores · decision threshold ${data.threshold>1?"above 100%":pct(data.threshold)}.`;
    frame.setAttribute("aria-busy","false");
    bindRecordDetailButtons(frame,context.run,view.model,context.hash);
  } catch(error) {if(request===fx.recordSeq&&fxCurrent(context)&&state.page==="records"){clearReport(false);showError(error.message);}}
}

async function fxDownload(url, filename, status) {
  const response=await fetch(url,{cache:"no-store"});
  if(!response.ok) {const data=await response.json().catch(()=>({error:"The export could not be read."}));throw new Error(data.error);}
  const blob=await response.blob(), objectURL=URL.createObjectURL(blob), link=document.createElement("a");
  link.href=objectURL;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(objectURL),1000);
  if(status) status.textContent="Exported the verified rows matching this view.";
}

function comparisonPage() {
  const choices=(fx.available||[]).filter(r=>!r.error);
  if(!choices.some(r=>r.id===fx.compareWith)) fx.compareWith=choices.find(r=>r.id!==state.run)?.id||state.run;
  const current=choices.find(r=>r.id===state.run)?.name||state.report.config.dashboard_name||"Selected run";
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>Compare two measured runs</h2><p class="panel-description">Left side: ${esc(current)}. Choose the right side below.</p></div><button id="compare-export" class="button primary" disabled>Export comparison</button></div><div class="comparison-picker"><label>Compare with<select id="compare-with">${fxOptions(choices.map(r=>[r.id,r.name]),fx.compareWith)}</select></label><p>Differences describe these saved runs. Different test records or attack settings are shown explicitly.</p></div><div id="comparison-body" aria-live="polite"><p class="no-jobs">Reading both verified runs…</p></div></section>`;
}
const fxPoints=v=>v==null?"—":`${v>0?"+":""}${(v*100).toFixed(1)} points`;
async function fxLoadComparison() {
  const context=fxContext(), right=fx.compareWith, request=++fx.compareSeq;
  fx.comparison=null; document.querySelector("#compare-export").disabled=true;
  document.querySelector("#comparison-body").innerHTML='<p class="no-jobs">Reading both verified runs…</p>';
  try {
    const data=await readJSON("/api/compare?"+fxQuery({left:context.run,right,left_sha:context.hash}));
    if(request!==fx.compareSeq||!fxCurrent(context)||state.page!=="compare") return;
    fx.comparison=data;
    const identity=r=>`<div><span>${esc(r.name)}</span><strong>${num(r.tested)} test records</strong><small>${num(r.attacked)} attacked · seed ${num(r.seed)}</small></div>`;
    document.querySelector("#comparison-body").innerHTML=`<div class="compare-identities">${identity(data.left)}${identity(data.right)}</div><div class="comparison-note">${data.same_run?"Both sides use the same saved run.":data.same_test_records?"Both runs use the same source test records.":"These runs use different test records. Read the figures as separate measurements."}</div><div class="table-wrap"><table><thead><tr><th>Filter</th><th>Spam caught · left</th><th>Spam caught · right</th><th>Change</th><th>False alarms · left</th><th>False alarms · right</th><th>F1 · left / right</th></tr></thead><tbody>${data.models.map(r=>`<tr><td class="method-name">${label(r.model)}</td><td>${pct(1-r.left.fnr)}</td><td>${pct(1-r.right.fnr)}</td><td>${fxPoints(r.change.spam_caught)}</td><td>${pct(r.left.fpr)}</td><td>${pct(r.right.fpr)}</td><td>${Number(r.left.f1).toFixed(3)} / ${Number(r.right.f1).toFixed(3)}</td></tr>`).join("")}</tbody></table></div><div class="panel-head comparison-subhead"><div><h2>What slipped through under attack</h2><p class="panel-description">${data.same_attacked_records?"Both runs attacked the same source records.":"The attacked records differ between these runs; these are separate measurements."} Counts and change limits are kept beside the results.</p></div></div><div class="table-wrap"><table><thead><tr><th>Attack / target</th><th>Escape rate · left</th><th>Escape rate · right</th><th>Change</th><th>Change limit · left / right</th><th>Initially caught · left / right</th></tr></thead><tbody>${data.attacks.map(r=>`<tr><td><span class="method-name">${esc(attackName(r)[0])}</span><span class="method-sub">${label(r.target)} · ${esc(attackName(r)[1])}</span></td><td>${pct(r.left.asr)}</td><td>${pct(r.right.asr)}</td><td>${fxPoints(r.escape_change)}</td><td>${r.left.epsilon==null?"Not capped":changeLevel(r.left.epsilon)} / ${r.right.epsilon==null?"Not capped":changeLevel(r.right.epsilon)}${r.same_limit?"":'<span class="method-sub">Different limits</span>'}</td><td>${num(r.left.eligible)} / ${num(r.right.eligible)}<span class="method-sub">${r.same_attacked_records?"Same attacked records":"Different attacked records"}</span></td></tr>`).join("")}</tbody></table></div><div class="table-foot">Changes are percentage-point differences, right minus left. Recorded processing times are included in the CSV; they vary with computer load.</div>`;
    document.querySelector("#compare-export").disabled=false;
  } catch(error) {if(request===fx.compareSeq&&fxCurrent(context)&&state.page==="compare") document.querySelector("#comparison-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;}
}

function fxOpenDialog(title) {
  document.querySelector("#feature-dialog-title").textContent=title;
  const dialog=document.querySelector("#feature-dialog"); if(!dialog.open) dialog.showModal();
  document.querySelector("#feature-dialog-body").innerHTML='<p class="no-jobs">Reading the saved training evidence…</p>';
}
function closeFeatureDialog() {const dialog=document.querySelector("#feature-dialog");if(dialog?.open) dialog.close();}
async function fxTrainingList(model, mode="selected", page=1) {
  const context=fxContext(), request=++fx.modalSeq;
  fx.training={model,mode,page};
  fxOpenDialog("Training records");
  try {
    const data=await readJSON("/api/training-records?"+fxQuery({run:context.run,report_sha256:context.hash,model,mode,page,size:25}));
    if(request!==fx.modalSeq||!fxCurrent(context)||!document.querySelector("#feature-dialog").open) return;
    fx.training.page=data.page;
    const modeNames={selected:"Selected for changes",modified:"Actually changed",removed:"Removed during cleanup",all:"All training records"};
    document.querySelector("#feature-dialog-body").innerHTML=`<div class="training-counts"><span>${num(data.counts.selected)} selected</span><span>${num(data.counts.modified)} changed</span><span>${num(data.counts.removed)} removed during cleanup</span></div><div class="inspector-controls"><label>Records to show<select id="training-mode">${fxOptions(Object.entries(modeNames),mode)}</select></label><span>${num(data.total)} matching records</span></div><div class="table-wrap"><table><thead><tr><th>Source row</th><th>Original label</th><th>Label after changes</th><th>Selected</th><th>Changed</th><th>Removed</th><th>Details</th></tr></thead><tbody>${data.rows.map(r=>`<tr><td>${num(r.source_row)}</td><td>${r.before_label?"Spam":"Regular email"}</td><td>${r.after_label?"Spam":"Regular email"}</td><td>${r.selected?"Yes":"No"}</td><td>${r.modified?"Yes":"No"}</td><td>${r.removed?"Yes":"No"}</td><td><button class="text-button training-record-open" data-position="${r.position}">Inspect record</button></td></tr>`).join("")}</tbody></table></div>${data.rows.length?"":'<p class="no-jobs">No records in this view.</p>'}<div class="page-controls"><span>Page ${num(data.page)} of ${num(data.pages)}</span><div class="row-actions"><button id="training-prev" class="button" ${data.page<=1?"disabled":""}>Previous</button><button id="training-next" class="button" ${data.page>=data.pages?"disabled":""}>Next</button></div></div><div class="table-foot">Removal marks come from the matching cleanup trial. ${data.scores_after_cleanup?"The selected model was trained after those removals.":"The selected model was trained before those removals."} Unchanged records may also have been removed.</div>`;
    document.querySelector("#training-mode").addEventListener("change",e=>fxTrainingList(model,e.target.value,1));
    document.querySelector("#training-prev").addEventListener("click",()=>fxTrainingList(model,mode,data.page-1));
    document.querySelector("#training-next").addEventListener("click",()=>fxTrainingList(model,mode,data.page+1));
    document.querySelectorAll(".training-record-open").forEach(b=>b.addEventListener("click",()=>fxTrainingRecord(model,Number(b.dataset.position))));
  } catch(error) {if(request===fx.modalSeq&&fxCurrent(context)) document.querySelector("#feature-dialog-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;}
}

async function fxTrainingRecord(model, position) {
  const context=fxContext(), request=++fx.modalSeq;
  fxOpenDialog("What changed in this training record");
  try {
    const data=await readJSON("/api/training-record?"+fxQuery({run:context.run,report_sha256:context.hash,model,position}));
    if(request!==fx.modalSeq||!fxCurrent(context)||!document.querySelector("#feature-dialog").open) return;
    const modelRow=state.report.poisoning.find(r=>r.model===model);
    const status=(p,t)=>`${p>=t?"Flagged":"Passed"} · ${pct(p)} spam score`;
    document.querySelector("#feature-dialog-body").innerHTML=`<div class="inspector-controls"><button id="training-back" class="button">Back to records</button><span>Source row ${num(data.source_row)}</span></div><div class="training-counts"><span>Original label: ${data.before_label?"Spam":"Regular email"}</span><span>After changes: ${data.after_label?"Spam":"Regular email"}</span><span>${data.removed?"Removed during cleanup":"Kept by cleanup"}</span></div><div class="table-wrap"><table><thead><tr><th>Filter</th><th>Original measurements</th><th>Changed measurements</th></tr></thead><tbody><tr><td>Original filter</td><td title="${data.clean_scores[0]}">${status(data.clean_scores[0],data.clean_threshold)}</td><td title="${data.clean_scores[1]}">${status(data.clean_scores[1],data.clean_threshold)}</td></tr><tr><td>${modelRow.sanitized?"After cleanup":"Changed training data"}</td><td title="${data.changed_scores[0]}">${status(data.changed_scores[0],data.changed_threshold)}</td><td title="${data.changed_scores[1]}">${status(data.changed_scores[1],data.changed_threshold)}</td></tr></tbody></table></div><div class="feature-heading"><p>${num(data.features.filter(r=>r.changed).length)} measurements changed.</p><label><input id="training-show-all" type="checkbox"> Show unchanged measurements</label></div><div id="training-features"></div><div class="table-foot">${data.scores_after_cleanup?"The selected model was trained after cleanup.":"Removal marks describe the matching cleanup trial; the selected model was trained before cleanup."} All four scores were calculated from the pinned model weights for this record. Before-and-after measurements are shown in the dataset's original units.</div>`;
    const draw=all=>{
      const rows=data.features.filter(r=>all||r.changed).sort((a,b)=>Math.abs(b.scaled_change)-Math.abs(a.scaled_change));
      document.querySelector("#training-features").innerHTML=rows.length?`<div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Before</th><th>After</th><th>Change</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(featureName(r.name))}${r.locked?'<span class="method-sub">Locked</span>':""}</td><td>${measurement(r.before)}</td><td>${measurement(r.after)}</td><td>${r.change>0?"+":""}${measurement(r.change)}</td></tr>`).join("")}</tbody></table></div>`:'<p class="no-jobs">The measurements stayed the same. Check the original and changed labels above.</p>';
    };
    draw(false);
    document.querySelector("#training-show-all").addEventListener("change",e=>draw(e.target.checked));
    document.querySelector("#training-back").addEventListener("click",()=>fxTrainingList(model,fx.training.mode,fx.training.page));
  } catch(error) {if(request===fx.modalSeq&&fxCurrent(context)) document.querySelector("#feature-dialog-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;}
}

function bindFeatureContent() {
  bindAnalysisContent();
  ++fx.recordSeq; ++fx.compareSeq;
  document.querySelectorAll(".inspect-training").forEach(b=>b.addEventListener("click",()=>fxTrainingList(b.dataset.model)));
  if(state.page==="records") {
    const change=(id,key,convert=v=>v)=>document.getElementById(id).addEventListener("change",e=>{fx.records[key]=convert(e.target.value);fx.records.page=1;fxLoadRecords();});
    change("records-model","model");change("records-label","label");change("records-outcome","outcome");change("records-sort","sort");change("records-size","size",Number);
    let timer;
    document.querySelector("#records-search").addEventListener("input",e=>{fx.records.search=e.target.value;fx.records.page=1;++fx.recordSeq;fxDisableRecordActions();clearTimeout(timer);timer=setTimeout(()=>{if(state.page==="records"&&state.report)fxLoadRecords();},250);});
    document.querySelector("#records-features").addEventListener("change",e=>fx.records.features=e.target.checked);
    document.querySelector("#records-clear").addEventListener("click",()=>{Object.assign(fx.records,{label:"all",outcome:"all",search:"",sort:"score-desc",page:1});for(const [id,key] of [["records-label","label"],["records-outcome","outcome"],["records-search","search"],["records-sort","sort"]]) document.getElementById(id).value=fx.records[key];fxLoadRecords();});
    document.querySelector("#records-recheck").addEventListener("click",()=>fxLoadRecords(true));
    document.querySelector("#records-prev").addEventListener("click",()=>{fx.records.page--;fxLoadRecords();});
    document.querySelector("#records-next").addEventListener("click",()=>{fx.records.page++;fxLoadRecords();});
    document.querySelector("#records-export").addEventListener("click",fxExportRecords);
    fxLoadRecords();
  }
  if(state.page==="compare") {
    document.querySelector("#compare-with").addEventListener("change",e=>{fx.compareWith=e.target.value;fxLoadComparison();});
    document.querySelector("#compare-export").addEventListener("click",fxExportComparison);
    fxLoadComparison();
  }
}

async function fxExportRecords(event) {
  const button=event.currentTarget, context=fxContext(), request=fx.recordSeq;
  const view={...fx.records};
  const current=()=>button.isConnected&&state.page==="records"&&fxCurrent(context)&&request===fx.recordSeq;
  button.disabled=true;
  try {
    await fxDownload("/api/records.csv?"+fxQuery({run:context.run,report_sha256:context.hash,...view,features:view.features?1:0}),"filter-records.csv");
    if(current()) document.querySelector("#records-status").textContent="Exported the verified rows matching this view.";
  } catch(error) {
    if(current()) {clearReport(false);showError(error.message);}
  } finally {if(current()) button.disabled=false;}
}

async function fxExportComparison(event) {
  const button=event.currentTarget, context=fxContext(), request=fx.compareSeq, data=fx.comparison;
  const current=()=>button.isConnected&&state.page==="compare"&&fxCurrent(context)&&request===fx.compareSeq&&data===fx.comparison;
  if(!data) return;
  button.disabled=true;
  try {
    await fxDownload("/api/compare.csv?"+fxQuery({left:data.left.id,right:data.right.id,left_sha:data.left.report_sha256,right_sha:data.right.report_sha256}),"run-comparison.csv");
  } catch(error) {
    if(current()) {
      document.querySelector("#comparison-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;
      fx.comparison=null;
    }
  } finally {if(current()) button.disabled=false;}
}

document.addEventListener("DOMContentLoaded",()=>{
  const dialog=document.querySelector("#feature-dialog");
  document.querySelector("#feature-dialog-close").addEventListener("click",()=>dialog.close());
  dialog.addEventListener("close",()=>{if(!dialog.open)++fx.modalSeq;});
  dialog.addEventListener("click",e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
});
