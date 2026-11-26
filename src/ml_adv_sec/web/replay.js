"use strict";

const replayState={sequence:0,otherSequence:0,result:null,other:null};
const replayOutcomes={"first-only":"Caught only in first","second-only":"Caught only in second","both-caught":"Caught in both","both-missed":"Missed in both"};
function replayAttackOptions(report,selected) {
  return fxOptions([...new Set(report.transferability.map(r=>r.attack))].map(id=>[id,attackName({attack:id})[1]]),selected);
}
function replayInvalidate() {
  ++replayState.sequence;replayState.result=null;
  for(const id of ["replay-export-models","replay-export-records","replay-prev","replay-next"])document.getElementById(id)?.setAttribute("disabled","");
  const body=document.querySelector("#replay-body");if(body)body.innerHTML="";
}
function bindAttackReplay() {
  replayInvalidate();++replayState.otherSequence;replayState.other=null;
  if(state.page!=="attacks"||!state.report)return;
  const available=(fx.available||[]).filter(r=>!r.error);
  if(!available.some(r=>r.id===state.run))available.push({id:state.run,name:"Selected run"});
  content.insertAdjacentHTML("afterbegin",`<section class="panel table-panel section-spacer" id="replay-panel"><div class="panel-head"><div><h2>Replay and compare saved attacks</h2><p class="panel-description">Run every trained filter on the first attack's saved records. Compare individual records with another attack or another run.</p></div></div><form id="replay-form"><div class="record-toolbar"><label>First attack<select id="replay-attack">${replayAttackOptions(state.report)}</select></label><label>Comparison run<select id="replay-other">${fxOptions(available.map(r=>[r.id,r.name]),state.run)}</select></label><label>Second attack<select id="replay-other-attack">${replayAttackOptions(state.report)}</select></label><label>Filter for paired records<select id="replay-model">${fxModelOptions("tree")}</select></label><label>Record outcome<select id="replay-category">${fxOptions([["all","All matching records"],["changed","Different after-attack decisions"],...Object.entries(replayOutcomes)],"all")}</select></label><label>Comparison rows per page<select id="replay-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],25)}</select></label></div><div class="record-actions"><button class="button primary" id="replay-run" type="submit">Run attack comparison</button><div class="row-actions"><button class="button" id="replay-export-models" type="button" disabled>Export filter replay</button><button class="button" id="replay-export-records" type="button" disabled>Export matching attack records</button></div></div></form><p id="replay-status" class="view-status" role="status">Choose two saved attacks and run the comparison. Both reports are checked before model execution.</p><div id="replay-body" aria-live="polite"></div><div class="page-controls"><span id="replay-range"></span><div class="row-actions"><button class="button" id="replay-prev" disabled>Previous attack records</button><button class="button" id="replay-next" disabled>Next attack records</button></div></div><div class="table-foot">These are fixed saved attack examples, not new attacks optimized against each filter. Cross-run decisions use each run's own model and threshold. Different training, scaling, and attack settings can all affect results. Only matching source rows are paired; unmatched rows are counted separately. Scores from different models are not equally calibrated.</div></section>`);
  replayState.other={report:state.report,hash:state.reportHash};
  const form=document.querySelector("#replay-form");
  form.addEventListener("submit",event=>{event.preventDefault();replayLoad(1);});
  form.addEventListener("change",()=>{replayInvalidate();document.querySelector("#replay-run").disabled=!replayState.other;document.querySelector("#replay-range").textContent="";document.querySelector("#replay-status").textContent="Settings changed. Run the comparison to calculate fresh results.";});
  document.querySelector("#replay-other").addEventListener("change",replayChooseOther);
  for(const scope of ["models","records"])document.getElementById("replay-export-"+scope).addEventListener("click",event=>replayExport(event,scope));
  for(const [id,offset] of [["prev",-1],["next",1]])document.getElementById("replay-"+id).addEventListener("click",()=>replayLoad(replayState.result.data.page+offset));
}
async function replayChooseOther() {
  const request=++replayState.otherSequence,context=fxContext(),run=document.querySelector("#replay-other").value;
  replayState.other=null;document.querySelector("#replay-run").disabled=true;
  try {
    const payload=run===context.run?{report:state.report,evidence:{report_sha256:context.hash}}:await readJSON("/api/report/"+encodeURIComponent(run));
    if(request!==replayState.otherSequence||!fxCurrent(context)||state.page!=="attacks")return;
    replayState.other={report:payload.report,hash:payload.evidence.report_sha256};
    document.querySelector("#replay-other-attack").innerHTML=replayAttackOptions(payload.report,document.querySelector("#replay-attack").value);
    const selector=document.querySelector("#replay-model"),selected=selector.value;
    const otherModels=new Set([...payload.report.clean,...payload.report.poisoning].map(r=>r.model));
    selector.innerHTML=fxOptions([...state.report.clean,...state.report.poisoning].filter(r=>otherModels.has(r.model)).map(r=>[r.model,fxModelName(r)]),selected);
    document.querySelector("#replay-status").textContent="Comparison report checked. Choose a filter available in both runs, then run the comparison.";
    document.querySelector("#replay-run").disabled=false;
  }catch(error){if(request===replayState.otherSequence&&fxCurrent(context)&&state.page==="attacks")document.querySelector("#replay-status").textContent=error.message;}
}
function replayQuery(page) {
  const value=id=>document.getElementById("replay-"+id).value;
  return {run:state.run,report_sha256:state.reportHash,attack:value("attack"),other:value("other"),other_sha256:replayState.other.hash,other_attack:value("other-attack"),model:value("model"),category:value("category"),size:Number(value("size")),page};
}
async function replayLoad(page) {
  if(!replayState.other)return;
  const context=fxContext(),query=replayQuery(page);replayInvalidate();const request=replayState.sequence;
  const button=document.querySelector("#replay-run");button.disabled=true;
  document.querySelector("#replay-status").textContent="Checking source records and running saved model weights on both attack sets…";
  try {
    const data=await readJSON("/api/attack-replay?"+fxQuery(query));
    if(request!==replayState.sequence||!fxCurrent(context)||state.page!=="attacks")return;
    replayState.result={data,query};
    document.querySelector("#replay-body").innerHTML=`<details class="analysis-details"><summary>First attack against all ${num(data.models.length)} trained filters</summary>${axTable(["Filter","Tested","Initially caught","Escaped","Escape rate","Caught after","Newly caught","Mean score movement"],data.models.map(r=>`<tr><td>${esc(fxModelName({model:r.model,...state.report.poisoning.find(p=>p.model===r.model)}))}</td><td>${num(r.tested)}</td><td>${num(r.initially_caught)}</td><td>${num(r.escaped)}</td><td>${pct(r.escape_rate)}</td><td>${num(r.caught_after)}</td><td>${num(r.newly_caught)}</td><td>${rxSigned(r.mean_score_change)}</td></tr>`).join(""))}</details><h3>Source-matched attack records</h3><div class="training-counts">${Object.entries(replayOutcomes).map(([key,name])=>`<span>${esc(name)}: ${num(data.counts[key])}</span>`).join("")}</div>${data.rows.length?axTable(["Source row","First score before → after","Second score before → after","After-attack outcome","Details"],data.rows.map(r=>`<tr><td>${num(r.source_row)}</td><td>${pct(r.first_before)} → ${pct(r.first_after)}</td><td>${pct(r.second_before)} → ${pct(r.second_after)}</td><td>${esc(replayOutcomes[r.outcome])}</td><td><button class="text-button replay-detail" data-source="${r.source_row}">Compare measurements</button></td></tr>`).join("")):'<p class="no-jobs">No source records match this view.</p>'}`;
    document.querySelector("#replay-status").textContent=`Executed ${num(data.models.length)} filters on ${num(data.first_records)} first-attack records. Paired ${num(data.counts.all)} shared source rows; ${num(data.first_unmatched)} first-only and ${num(data.second_unmatched)} second-only rows were excluded. Escape rates use each filter's initially caught records; a dash means none were eligible.`;
    document.querySelector("#replay-range").textContent=`${num(data.total)} matching records · page ${num(data.page)} of ${num(data.pages)}`;
    document.querySelector("#replay-prev").disabled=data.page<=1;document.querySelector("#replay-next").disabled=data.page>=data.pages;
    for(const scope of ["models","records"])document.getElementById("replay-export-"+scope).disabled=false;
    document.querySelectorAll(".replay-detail").forEach(item=>item.addEventListener("click",()=>replayDetail(query,Number(item.dataset.source))));
  }catch(error){if(request===replayState.sequence&&fxCurrent(context)&&state.page==="attacks")document.querySelector("#replay-status").textContent=error.message;}
  finally{if(button.isConnected&&request===replayState.sequence)button.disabled=false;}
}
async function replayExport(event,scope) {
  const result=replayState.result;if(!result)return;const button=event.currentTarget;button.disabled=true;
  try{await fxDownload("/api/attack-replay.csv?"+fxQuery({...result.query,scope}),`attack-replay-${scope}.csv`);}
  catch(error){if(button.isConnected)document.querySelector("#replay-status").textContent=error.message;}
  finally{if(button.isConnected&&result===replayState.result)button.disabled=false;}
}
async function replayDetail(query,source) {
  const request=++fx.modalSeq;fxOpenDialog("Attack measurement comparison");
  const params={...query,source_row:source};
  try {
    const data=await readJSON("/api/attack-replay-record?"+fxQuery(params));
    if(request!==fx.modalSeq||!document.querySelector("#feature-dialog").open)return;
    document.querySelector("#feature-dialog-body").innerHTML=`<p class="panel-description">Source row ${num(source)} · scores recalculated from the saved model weights. Values below are in original measurement units, reconstructed with each run's verified training scale.</p><div class="training-counts"><span>First threshold: ${pct(data.first_threshold)}</span><span>Second threshold: ${pct(data.second_threshold)}</span><span>${esc(replayOutcomes[data.record.outcome])}</span></div><div class="record-actions"><button class="button primary" id="replay-export-changes">Export all 57 measurement changes</button></div>${axTable(["Measurement","First original → changed","First change","Second original → changed","Second change","Locked"],data.changes.map(r=>`<tr><td>${esc(featureName(r.name))}</td><td>${measurement(r.first_original)} → ${measurement(r.first_changed)}</td><td>${measurement(r.first_change)}</td><td>${measurement(r.second_original)} → ${measurement(r.second_changed)}</td><td>${measurement(r.second_change)}</td><td>${r.first_locked?"First":"—"} / ${r.second_locked?"Second":"—"}</td></tr>`).join(""))}<p id="replay-detail-status" class="view-status" role="status">Measurement changes describe these saved examples; they do not assign a causal contribution to the prediction.</p>`;
    document.querySelector("#replay-export-changes").addEventListener("click",async event=>{const button=event.currentTarget;button.disabled=true;try{await fxDownload("/api/attack-replay.csv?"+fxQuery({...params,scope:"changes"}),"attack-measurement-changes.csv");}catch(error){if(button.isConnected)document.querySelector("#replay-detail-status").textContent=error.message;}finally{if(button.isConnected)button.disabled=false;}});
  }catch(error){if(request===fx.modalSeq&&document.querySelector("#feature-dialog").open)document.querySelector("#feature-dialog-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;}
}
