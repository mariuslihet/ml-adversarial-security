"use strict";

const rx={importance:{model:"clean",partition:"test",repeats:3,seed:42,sort:"brier_increase"},importanceSeq:0,importanceResult:null,overlap:{target:"clean",category:"all",page:1,size:25},overlapSeq:0,overlapResult:null};
const rxMetricNames={brier_increase:"Prediction error increase",accuracy_drop:"Accuracy drop",recall_drop:"Spam caught drop",false_alarm_change:"False-alarm change",mean_score_change:"Mean score movement",changed_decisions:"Decisions changed"};
const rxSigned=value=>`${value>0?"+":""}${Number(value).toFixed(6)}`;
function rxValue(key,value,signed=true) {
  const prefix=signed&&value>0?"+":"";
  if(["accuracy_drop","recall_drop","false_alarm_change"].includes(key))return `${prefix}${Number(value*100).toFixed(3)} points`;
  return prefix+Number(value).toFixed(key==="changed_decisions"?2:6);
}
const rxAttack=id=>attackName({attack:id})[1];

function importancePage() {
  if(![...state.report.clean,...state.report.poisoning].some(r=>r.model===rx.importance.model))rx.importance.model="clean";
  const draft=rx.importance;
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>Which measurements does this filter rely on?</h2><p class="panel-description">Shuffle one measurement among actual records, rerun the saved model, and measure what changes. Every repeat runs the model again.</p></div><button id="importance-export" class="button primary" disabled>Export every repeat</button></div><form id="importance-form"><div class="record-toolbar"><label>Filter<select id="importance-model">${axModelOptions(state.report,draft.model)}</select></label><label>Records<select id="importance-partition">${fxOptions([["test","Held-out test records"],["val","Validation records"]],draft.partition)}</select></label><label>Repeats per measurement<input id="importance-repeats" type="number" min="1" max="10" step="1" required value="${draft.repeats}"></label><label>Repeatable shuffle seed<input id="importance-seed" type="number" min="0" max="2147483647" step="1" required value="${draft.seed}"></label></div><div class="record-actions"><label>Order measured results by<select id="importance-sort">${fxOptions(Object.entries(rxMetricNames),draft.sort)}</select></label><button class="button primary" type="submit">Run measurement checks</button></div></form><p id="importance-status" class="view-status" role="status">Choose a model and run the checks. Three repeats evaluate all 57 measurements 171 times.</p><div id="importance-body" aria-live="polite"></div><div class="table-foot">Values come from actual records. Shuffling breaks relationships between measurements, so this measures model dependence, not cause or attack success. Locked attack fields are included in this diagnostic. Negative changes are kept as measured.</div></section>`;
}

function rxDrawImportance() {
  const result=rx.importanceResult;if(!result)return;
  const data=result.data,key=rx.importance.sort;
  const rows=[...data.rows].sort((a,b)=>b.mean[key]-a.mean[key]||a.column-b.column);
  document.querySelector("#importance-body").innerHTML=`<div class="confusion-grid"><div><span>Records checked each time</span><strong>${num(data.records)}</strong><small>${data.partition==="test"?"Held-out test":"Validation"} partition</small></div><div><span>Model checks after shuffling</span><strong>${num(data.feature_checks)}</strong><small>${num(data.repeats)} repeats × 57 measurements</small></div><div><span>Original prediction error</span><strong>${Number(data.baseline.brier).toFixed(5)}</strong><small>Mean squared score error</small></div><div><span>Original spam caught</span><strong>${pct(data.baseline.recall)}</strong><small>${num(data.baseline.caught)} of ${num(data.baseline.caught+data.baseline.missed)} spam records</small></div></div>${axTable(["Measurement",`${rxMetricNames[key]} · mean ± spread`,"Spam caught drop","False-alarm change","Mean decisions changed","Details"],rows.map(r=>`<tr><td class="method-name">${esc(featureName(r.feature))}<span class="method-sub">${num(r.distinct_values)} distinct values${r.locked?" · locked during attacks":""}</span></td><td title="${r.mean[key]}">${rxValue(key,r.mean[key])} ± ${rxValue(key,r.spread[key],false)}</td><td>${fxPoints(r.mean.recall_drop)}</td><td>${fxPoints(r.mean.false_alarm_change)}</td><td>${Number(r.mean.changed_decisions).toFixed(2)}</td><td><button class="text-button importance-trials" data-column="${r.column}">Inspect repeats</button></td></tr>`).join(""))}<div class="table-foot">Spread is the sample standard deviation across repeats; one repeat has no measured spread. Larger error increases indicate greater reliance under this shuffle method. This is not a confidence interval.</div>`;
  document.querySelectorAll(".importance-trials").forEach(button=>button.addEventListener("click",()=>{
    const row=data.rows.find(r=>r.column===Number(button.dataset.column));
    ++fx.modalSeq;fxOpenDialog(featureName(row.feature));
    document.querySelector("#feature-dialog-body").innerHTML=`<div class="training-counts"><span>${esc(fxModelName([...state.report.clean,...state.report.poisoning].find(r=>r.model===data.model)))}</span><span>${num(data.records)} actual records</span><span>Seed ${num(data.seed)}</span></div>${axTable(["Repeat","Error increase","Accuracy drop","Spam caught drop","False-alarm change","Mean score movement","Decisions changed"],row.trials.map(r=>`<tr><td>${num(r.repeat)}</td><td>${rxSigned(r.brier_increase)}</td><td>${fxPoints(r.accuracy_drop)}</td><td>${fxPoints(r.recall_drop)}</td><td>${fxPoints(r.false_alarm_change)}</td><td>${Number(r.mean_score_change).toFixed(6)}</td><td>${num(r.changed_decisions)}</td></tr>`).join(""))}<div class="table-foot">Each row is a completed model run with this column shuffled. Original values were reassigned among the selected partition's records; no new measurement values were invented.</div>`;
  }));
}

async function rxLoadImportance() {
  const context=fxContext(),request=++rx.importanceSeq,view={...rx.importance};
  const query={run:context.run,report_sha256:context.hash,model:view.model,partition:view.partition,repeats:view.repeats,seed:view.seed};
  rx.importanceResult=null;document.querySelector("#importance-export").disabled=true;
  document.querySelector("#importance-status").textContent=`Running ${num(57*view.repeats)} measurement checks against the pinned model…`;
  document.querySelector("#importance-body").innerHTML="";
  try {
    const data=await readJSON("/api/importance?"+fxQuery(query));
    if(request!==rx.importanceSeq||!fxCurrent(context)||state.page!=="importance")return;
    rx.importanceResult={data,query};rxDrawImportance();
    document.querySelector("#importance-status").textContent=`Completed ${num(data.feature_checks)} actual model checks. Largest difference from saved test scores before shuffling: ${data.maximum_score_difference.toExponential(2)}. Export repeats the calculation using these exact settings.`;
    document.querySelector("#importance-export").disabled=false;
  }catch(error){if(request===rx.importanceSeq&&fxCurrent(context)&&state.page==="importance")document.querySelector("#importance-status").textContent=error.message;}
}

function overlapPage() {
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>Do different attacks fool the same records?</h2><p class="panel-description">Compare saved attacks on their shared source records. Scores are recalculated from the pinned target model before any result is shown.</p></div><button id="overlap-export" class="button primary" disabled>Export record outcomes</button></div><div class="record-toolbar"><label>Target filter<select id="overlap-target">${fxOptions([["clean","Original filter"],["tree","Tree filter"]],rx.overlap.target)}</select></label><label>Records to show<select id="overlap-category">${fxOptions([["all","Every shared record"],["escaped","Escaped at least one attack"],["resisted","Caught after every attack"],["one","Escaped exactly one attack"]],rx.overlap.category)}</select></label><label>Rows per page<select id="overlap-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],rx.overlap.size)}</select></label><div class="research-note">Only spam caught before an attack counts toward escape rates. Originally missed spam is shown separately.</div></div><p id="overlap-status" class="view-status" role="status">Checking saved attacks and rerunning the target model…</p><div id="overlap-body" aria-live="polite"></div><div class="page-controls"><span id="overlap-range"></span><div class="row-actions"><button id="overlap-prev" class="button" disabled>Previous</button><button id="overlap-next" class="button" disabled>Next</button></div></div></section>`;
}

async function rxLoadOverlap() {
  const context=fxContext(),request=++rx.overlapSeq,view={...rx.overlap};
  const query={run:context.run,report_sha256:context.hash,...view};
  rx.overlapResult=null;for(const id of ["overlap-export","overlap-prev","overlap-next"])document.getElementById(id).disabled=true;
  document.querySelector("#overlap-status").textContent="Verifying attack records and recalculating before-and-after scores…";
  document.querySelector("#overlap-body").innerHTML="";document.querySelector("#overlap-range").textContent="";
  try {
    const data=await readJSON("/api/attack-overlap?"+fxQuery(query));
    if(request!==rx.overlapSeq||!fxCurrent(context)||state.page!=="overlap")return;
    rx.overlap.page=data.page;rx.overlapResult={data,query};
    const headers=data.attacks.map(r=>`${esc(rxAttack(r.attack))}<span class="method-sub">${num(r.escaped)} escapes · ${pct(r.escape_rate)}</span>`);
    document.querySelector("#overlap-body").innerHTML=`<div class="confusion-grid">${[["Shared attack records",data.matched],["Caught before attacks",data.counts.eligible],["Escaped at least one",data.counts.escaped],["Already missed before attacks",data.counts.already_missed]].map(([name,count])=>`<div><span>${name}</span><strong>${num(count)}</strong><small>Actual shared source records</small></div>`).join("")}</div><div class="panel-head analysis-subhead"><div><h2>Shared escapes between attacks</h2><p class="panel-description">Each cell counts records that escaped both attacks. Overlap share divides that count by records escaping either attack; no escapes means the share is undefined.</p></div></div>${axTable(["Attack",...data.attacks.map(r=>esc(rxAttack(r.attack)))],data.attacks.map(first=>`<tr><th scope="row">${esc(rxAttack(first.attack))}</th>${data.attacks.map(second=>{const pair=data.overlaps.find(r=>r.left===first.attack&&r.right===second.attack);return `<td><strong>${num(pair.both)}</strong><span class="method-sub">${pct(pair.jaccard)} overlap share</span></td>`;}).join("")}</tr>`).join(""))}<div class="panel-head analysis-subhead"><div><h2>Per-record attack outcomes</h2><p class="panel-description">Open any score to inspect that exact saved attack. CSV export contains one row per matching source record and attack.</p></div></div>${data.rows.length?axTable(["Source row","Attacks escaped",...headers],data.rows.map(r=>`<tr><td class="method-name">${num(r.source_row)}${r.eligible?"":'<span class="method-sub">Missed before attacks</span>'}</td><td>${num(r.escaped_count)}</td>${r.attacks.map(a=>`<td><button class="text-button overlap-inspect" data-attack="${esc(a.attack)}" data-sample="${a.sample}">${pct(a.score,3)}</button><span class="method-sub">${r.eligible?(a.escaped?"Escaped":"Caught"):(a.flagged?"Caught after attack":"Still missed")}</span></td>`).join("")}</tr>`).join("")):'<p class="no-jobs">No shared records match this view.</p>'}<div class="table-foot">Attack settings differ, so these figures are not a ranking at equal strength. Excluded records per attack: ${data.attacks.map(a=>`${esc(rxAttack(a.attack))}: ${num(a.excluded)}`).join(" · ")}. A record already missed before an attack never counts as an attack escape.</div>`;
    document.querySelector("#overlap-status").textContent=`Before-and-after model scores match the saved evaluation. Decision threshold: ${data.threshold>1?"Above 100%":pct(data.threshold,4)}.`;
    document.querySelector("#overlap-range").textContent=`${num(data.total)} matching records · page ${num(data.page)} of ${num(data.pages)}`;
    document.querySelector("#overlap-prev").disabled=data.page<=1;document.querySelector("#overlap-next").disabled=data.page>=data.pages;document.querySelector("#overlap-export").disabled=false;
    document.querySelectorAll(".overlap-inspect").forEach(button=>button.addEventListener("click",()=>inspectAttack(button.dataset.attack,Number(button.dataset.sample),view.target)));
  }catch(error){if(request===rx.overlapSeq&&fxCurrent(context)&&state.page==="overlap")document.querySelector("#overlap-status").textContent=error.message;}
}

function bindResearchContent() {
  ++rx.importanceSeq;++rx.overlapSeq;rx.importanceResult=null;rx.overlapResult=null;
  if(state.page==="importance") {
    const form=document.querySelector("#importance-form");
    const sync=()=>{for(const key of ["model","partition","repeats","seed"])rx.importance[key]=["repeats","seed"].includes(key)?Number(document.getElementById("importance-"+key).value):document.getElementById("importance-"+key).value;};
    form.addEventListener("input",event=>{if(event.target.id==="importance-sort")return;sync();++rx.importanceSeq;rx.importanceResult=null;document.querySelector("#importance-export").disabled=true;document.querySelector("#importance-body").innerHTML="";document.querySelector("#importance-status").textContent="Settings changed. Run measurement checks to calculate these results.";});
    form.addEventListener("submit",event=>{event.preventDefault();sync();if(form.reportValidity())rxLoadImportance();});
    document.querySelector("#importance-sort").addEventListener("change",event=>{rx.importance.sort=event.target.value;rxDrawImportance();});
    document.querySelector("#importance-export").addEventListener("click",event=>rxExport(event,"importance"));
  }
  if(state.page==="overlap") {
    for(const key of ["target","category","size"])document.getElementById("overlap-"+key).addEventListener("change",event=>{rx.overlap[key]=key==="size"?Number(event.target.value):event.target.value;rx.overlap.page=1;rxLoadOverlap();});
    document.querySelector("#overlap-prev").addEventListener("click",()=>{--rx.overlap.page;rxLoadOverlap();});
    document.querySelector("#overlap-next").addEventListener("click",()=>{++rx.overlap.page;rxLoadOverlap();});
    document.querySelector("#overlap-export").addEventListener("click",event=>rxExport(event,"overlap"));
    rxLoadOverlap();
  }
}

async function rxExport(event,kind) {
  const button=event.currentTarget,result=rx[kind+"Result"],context=fxContext(),sequence=rx[kind+"Seq"];
  if(!result)return;button.disabled=true;
  const current=()=>button.isConnected&&fxCurrent(context)&&sequence===rx[kind+"Seq"]&&result===rx[kind+"Result"];
  try {await fxDownload((kind==="importance"?"/api/importance.csv?":"/api/attack-overlap.csv?")+fxQuery(result.query),kind==="importance"?"input-importance.csv":"attack-overlap.csv");}
  catch(error){if(current()){document.querySelector(kind==="importance"?"#importance-status":"#overlap-status").textContent=error.message;rx[kind+"Result"]=null;}}
  finally{if(current())button.disabled=false;}
}
