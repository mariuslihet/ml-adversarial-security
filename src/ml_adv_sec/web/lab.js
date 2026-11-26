"use strict";

const ml={tool:"ensemble",sequence:0,result:null,meta:null,page:1,outcome:"all",trial:"",band:null,drafts:{
  ensemble:{members:["clean","tree","pgd"],mode:"fit",fpr:1,attack:""},
  calibration:{model:"tree",method:"sigmoid",fpr:1,bins:10},
  stress:{model:"clean",columns:[0,1,2],replacement:"median"}
}};
const mlNames={ensemble:"Combine filters",calibration:"Calibrate confidence",stress:"Remove measurements"};
const mlMetricNames={recall:"Spam caught",fpr:"False alarms",precision:"Precision",f1:"F1",brier:"Score error",log_loss:"Log loss",accuracy:"Accuracy"};
const mlMetric=(key,value)=>["recall","fpr","precision","accuracy"].includes(key)?pct(value):value==null?"—":Number(value).toFixed(5);
const mlThreshold=value=>value>1?"Above 100%":pct(value,3);
const mlRows=()=>[...state.report.clean,...state.report.poisoning];
const mlName=model=>{const row=mlRows().find(r=>r.model===model);return row?fxModelName(row):model;};
function mlChoices(items,chosen,name) {
  return `<div class="model-lab-choices">${items.map(([value,title])=>`<label><input type="checkbox" name="${name}" value="${esc(value)}" ${chosen.includes(value)?"checked":""}><span>${esc(title)}</span></label>`).join("")}</div>`;
}
function modelLabPage() {
  const draft=ml.drafts[ml.tool],ids=new Set(mlRows().map(r=>r.model));
  if(ml.tool==="ensemble")draft.members=draft.members.filter(id=>ids.has(id));
  else if(!ids.has(draft.model))draft.model="clean";
  let fields="",description="",action="";
  if(ml.tool==="ensemble") {
    description="Fit a weighted combination of trained filters, then measure it on untouched test records and a saved attack.";
    fields=`<div class="record-toolbar"><label>Weight method<select name="mode">${fxOptions([["fit","Fit weights on validation records"],["equal","Use equal shares"]],draft.mode)}</select></label><label>False-alarm target (%)<input name="fpr" type="number" min="0" max="25" step="any" required value="${draft.fpr}"></label><label>Saved attack to replay<select name="attack">${fxOptions([["","Do not replay an attack"],...[...new Set(state.report.transferability.map(r=>r.attack))].map(id=>[id,attackName({attack:id})[1]])],draft.attack)}</select></label></div><details class="analysis-details" open><summary>Choose 2–8 trained filters</summary>${mlChoices(mlRows().map(r=>[r.model,fxModelName(r)]),draft.members,"members")}</details>`;
    action="Fit and test ensemble";
  }else if(ml.tool==="calibration") {
    description="Fit a correction to a filter's confidence scores and check whether they better match observed spam frequencies on test records.";
    fields=`<div class="record-toolbar"><label>Filter<select name="model">${axModelOptions(state.report,draft.model)}</select></label><label>Correction method<select name="method">${fxOptions([["sigmoid","Fitted sigmoid"],["isotonic","Monotone fitted curve"]],draft.method)}</select></label><label>False-alarm target (%)<input name="fpr" type="number" min="0" max="25" step="any" required value="${draft.fpr}"></label><label>Confidence bands<select name="bins">${fxOptions([[5,"5"],[10,"10"],[20,"20"]],draft.bins)}</select></label></div>`;
    action="Fit and test calibration";
  }else {
    description="Replace selected measurements with training medians or zero, then rerun the model. Check fields separately and together in the chosen column order.";
    fields=`<div class="record-toolbar"><label>Filter<select name="model">${axModelOptions(state.report,draft.model)}</select></label><label>Unavailable-value rule<select name="replacement">${fxOptions([["median","Training median"],["zero","Zero in original units"]],draft.replacement)}</select></label></div><details class="analysis-details" open><summary>Choose 1–12 measurements</summary>${ml.meta?.report_sha256===state.reportHash?mlChoices(ml.meta.measurements.map(r=>[r.column,featureName(r.name)+(r.locked_for_attacks?" · locked for attacks":"")]),draft.columns,"columns"):'<p class="no-jobs">Reading the verified measurement names…</p>'}</details>`;
    action="Run measurement stress test";
  }
  return `<section class="panel table-panel model-lab-panel"><div class="panel-head"><div><h2>${mlNames[ml.tool]}</h2><p class="panel-description">Work with the selected run's saved filters and email records.</p></div></div><div class="model-lab-switch" role="group" aria-label="Model tools">${Object.entries(mlNames).map(([key,name])=>`<button class="button ${key===ml.tool?"primary":""}" data-lab-tool="${key}" aria-pressed="${key===ml.tool}">${name}</button>`).join("")}</div><p class="comparison-note">${description}</p><form id="model-lab-form">${fields}<div class="record-actions"><button class="button primary" id="model-lab-run" type="submit" ${ml.tool==="stress"&&ml.meta?.report_sha256!==state.reportHash?"disabled":""}>${action}</button></div></form><p id="model-lab-status" class="view-status" role="status">Run the tool to calculate measured results. Earlier run files are retained.</p><div id="model-lab-results" aria-live="polite"></div><div class="table-foot">${ml.tool==="stress"?"Replacement values come from training records or an explicit raw value of zero. This is an input-availability diagnostic; locked attack fields can be tested here. The saved decision threshold stays fixed.":"Fitting and threshold selection use separate halves of the validation partition. Test records measure the result and never fit parameters. Each model comparison uses the same threshold-selection records and false-alarm target. Improvements are measured, not guaranteed."}</div></section>`;
}
function mlReadForm() {
  const form=document.querySelector("#model-lab-form"),draft=ml.drafts[ml.tool],values=new FormData(form);
  for(const key of Object.keys(draft))draft[key]=key==="members"?values.getAll(key):key==="columns"?values.getAll(key).map(Number):["fpr","bins"].includes(key)?Number(values.get(key)):values.get(key);
}
function mlInvalidate() {
  ++ml.sequence;ml.result=null;ml.band=null;ml.page=1;ml.outcome="all";
  const result=document.querySelector("#model-lab-results");if(result)result.innerHTML="";
}
function bindModelLab() {
  mlInvalidate();
  if(state.page!=="lab")return;
  document.querySelectorAll("[data-lab-tool]").forEach(button=>button.addEventListener("click",()=>{mlReadForm();ml.tool=button.dataset.labTool;render();}));
  const form=document.querySelector("#model-lab-form");
  form.addEventListener("change",()=>{mlReadForm();mlInvalidate();document.querySelector("#model-lab-run").disabled=ml.tool==="stress"&&ml.meta?.report_sha256!==state.reportHash;document.querySelector("#model-lab-status").textContent="Settings changed. Run the tool to calculate fresh results.";});
  form.addEventListener("submit",event=>{event.preventDefault();if(form.reportValidity()){mlReadForm();mlRun();}});
  if(ml.tool==="stress"&&ml.meta?.report_sha256!==state.reportHash) {
    const context=fxContext();
    readJSON("/api/model-lab-options?"+fxQuery({run:context.run,report_sha256:context.hash})).then(data=>{if(fxCurrent(context)){const changed=ml.meta?.report_sha256!==data.report_sha256;ml.meta=data;if(changed&&state.page==="lab"&&ml.tool==="stress")render();}}).catch(error=>{if(state.page==="lab"&&ml.tool==="stress"&&fxCurrent(context))document.querySelector("#model-lab-status").textContent=error.message;});
  }
}
async function mlRun() {
  const draft=ml.drafts[ml.tool],context=fxContext();mlInvalidate();
  if(ml.tool==="ensemble"&&(draft.members.length<2||draft.members.length>8)||ml.tool==="stress"&&(draft.columns.length<1||draft.columns.length>12)) {
    document.querySelector("#model-lab-status").textContent=ml.tool==="ensemble"?"Select two to eight filters.":"Select one to twelve measurements.";return;
  }
  const sequence=ml.sequence,query={run:context.run,report_sha256:context.hash,tool:ml.tool,...draft};
  if(query.members)query.members=query.members.join(",");if(query.columns)query.columns=query.columns.join(",");if("fpr" in query)query.fpr/=100;
  const button=document.querySelector("#model-lab-run");button.disabled=true;
  document.querySelector("#model-lab-status").textContent="Checking the saved evidence, running model weights, and measuring held-out records…";
  try {
    const data=await readJSON("/api/model-lab?"+fxQuery(query));
    if(sequence!==ml.sequence||!fxCurrent(context)||state.page!=="lab")return;
    document.querySelector("#model-lab-form details")?.removeAttribute("open");
    ml.result={data,query};ml.trial=data.trials?.at(-1).id||"";mlDraw();
    document.querySelector("#model-lab-status").textContent=data.tool==="stress"?`Completed ${num(data.trials.length)} real model trials on ${num(data.test_records)} test records.`:data.tool==="ensemble"&&query.mode==="equal"?`Combined filters with equal shares, selected thresholds on ${num(data.split.threshold_source_rows.length)} validation records, and tested ${num(data.test_records)} untouched records.`:`Fitted on ${num(data.split.fit_source_rows.length)} validation records, selected thresholds on ${num(data.split.threshold_source_rows.length)} separate validation records, and tested ${num(data.test_records)} untouched records.`;
  }catch(error){if(sequence===ml.sequence&&fxCurrent(context)&&state.page==="lab")document.querySelector("#model-lab-status").textContent=error.message;}
  finally{if(button.isConnected&&sequence===ml.sequence)button.disabled=false;}
}
function mlMetricsTable(items) {
  return axTable(["Measured test result",...items.map(([name])=>esc(name))],Object.entries(mlMetricNames).map(([key,name])=>`<tr><th scope="row">${name}</th>${items.map(([,metrics])=>`<td>${mlMetric(key,metrics[key])}</td>`).join("")}</tr>`).join(""));
}
function mlDraw() {
  const data=ml.result.data;
  let body="";
  if(data.tool==="ensemble") {
    body=`<h3>Ensemble test result</h3>${mlMetricsTable([["Ensemble",data.metrics]])}<h3>Members at the same false-alarm target</h3>${axTable(["Filter","Fitted share","Threshold","Spam caught","False alarms","F1","Score error"],data.members.map(r=>`<tr><td>${esc(mlName(r.model))}</td><td>${pct(r.weight,2)}</td><td>${mlThreshold(r.threshold)}</td><td>${pct(r.metrics.recall)}</td><td>${pct(r.metrics.fpr)}</td><td>${Number(r.metrics.f1).toFixed(5)}</td><td>${Number(r.metrics.brier).toFixed(5)}</td></tr>`).join(""))}<p class="comparison-note">Fitting score error: ${data.fitting.uniform_brier.toFixed(6)} with equal shares; ${data.fitting.fitted_brier.toFixed(6)} with the selected method. Threshold ${mlThreshold(data.threshold)}. Member thresholds are selected again on the same calibration records as the ensemble.</p>${data.attack_metrics?`<h3>Ensemble on the saved attack</h3><div class="training-counts"><span>${num(data.attack_metrics.records)} attacked records</span><span>${num(data.attack_metrics.initially_caught)} initially caught</span><span>${num(data.attack_metrics.escaped)} escaped</span><span>Escape rate: ${pct(data.attack_metrics.escape_rate)}</span><span>${num(data.attack_metrics.caught_after)} caught afterward</span></div><p class="comparison-note">Fixed saved examples are replayed through the ensemble. This is not a new attack optimized against the combined filter.</p>`:""}`;
  }else if(data.tool==="calibration") {
    body=`<h3>Before and after confidence correction</h3>${mlMetricsTable([["Original scores",data.before],["Corrected scores",data.after]])}<div class="training-counts"><span>Original confidence error: ${pct(data.before_reliability.expected_error,2)}</span><span>Corrected confidence error: ${pct(data.after_reliability.expected_error,2)}</span></div><p class="comparison-note">Confidence error is the record-weighted absolute gap between mean score and observed spam share in the chosen bands. Both decision thresholds are chosen using the separate threshold-selection records. A monotone correction can change score accuracy without improving classification.</p><h3>Confidence bands on test records</h3>${axTable(["Scores","Band","Records","Mean score","Observed spam","Gap","Details"],["before","after"].map(version=>data[version+"_reliability"].bins.map(r=>`<tr><td>${version==="before"?"Original":"Corrected"}</td><td>${pct(r.low,0)}–${pct(r.high,0)}</td><td>${num(r.count)}</td><td>${pct(r.mean_score)}</td><td>${pct(r.spam_share)}</td><td>${pct(r.gap)}</td><td><button class="text-button model-lab-band" data-version="${version}" data-bin="${r.bin}" ${r.count?"":"disabled"} aria-label="Inspect ${version} confidence band ${r.bin+1}">Inspect band</button></td></tr>`).join("")).join(""))}`;
  }else {
    body=`<p class="comparison-note">Original test result: ${pct(data.baseline.recall)} spam caught, ${pct(data.baseline.fpr)} false alarms, and score error ${data.baseline.brier.toFixed(5)}. All trials keep the same threshold.</p><h3>Single measurements and cumulative losses</h3>${axTable(["Trial","Measurements removed","New mistakes","Mistakes fixed","Spam caught","False alarms","Score error","Details"],data.trials.map(r=>`<tr><td>${r.id.startsWith("single")?"Single field":"Cumulative step "+r.id.split("-")[1]}</td><td>${r.columns.map(i=>esc(featureName(ml.meta.measurements[i].name))).join(", ")}</td><td>${num(r.new_mistakes)}</td><td>${num(r.fixed_mistakes)}</td><td>${pct(r.metrics.recall)}</td><td>${pct(r.metrics.fpr)}</td><td>${r.metrics.brier.toFixed(5)}</td><td><button class="text-button model-lab-trial" data-trial="${r.id}" aria-label="Inspect ${r.id} records">Inspect records</button></td></tr>`).join(""))}<details class="analysis-details"><summary>Exact replacement values</summary>${axTable(["Measurement","Value in original units","Scaled model input","Locked in attacks"],data.replacements.map(r=>`<tr><td>${esc(featureName(r.name))}</td><td>${measurement(r.raw_value)}</td><td>${measurement(r.scaled_value)}</td><td>${r.locked_for_attacks?"Yes":"No"}</td></tr>`).join(""))}</details>`;
  }
  document.querySelector("#model-lab-results").innerHTML=`${body}<div class="record-actions model-lab-exports"><div class="row-actions"><button class="button" data-lab-export="summary">Export measured summary</button><button class="button" data-lab-export="records">${data.tool==="stress"?"Export selected trial records":"Export all test records"}</button>${data.tool==="calibration"?'<button class="button" data-lab-export="bands">Export confidence bands</button>':""}${data.attack_metrics?'<button class="button" data-lab-export="attacks">Export ensemble attack records</button>':""}<button class="button primary" data-lab-export="recipe">${data.tool==="stress"?"Export stress-test recipe":"Export fitted recipe"}</button></div></div><h3>Inspect test records</h3><div class="record-toolbar"><label>Record result<select id="model-lab-outcome">${fxOptions([["all","All records"],["mistakes","Current mistakes"],...(data.tool==="ensemble"?[]:[["changed","Changed decisions"],["new-errors","New mistakes"],["fixed-errors","Fixed mistakes"]])],"all")}</select></label><button class="button" id="model-lab-clear-band" ${data.tool==="calibration"?"":"hidden"}>Clear confidence-band filter</button></div><p id="model-lab-record-status" class="view-status" role="status"></p><div id="model-lab-records"></div><div class="page-controls"><span id="model-lab-range"></span><div class="row-actions"><button class="button" id="model-lab-prev">Previous test records</button><button class="button" id="model-lab-next">Next test records</button></div></div>`;
  document.querySelectorAll("[data-lab-export]").forEach(button=>button.addEventListener("click",mlExport));
  document.querySelectorAll(".model-lab-band").forEach(button=>button.addEventListener("click",()=>{ml.band={version:button.dataset.version,bin:Number(button.dataset.bin)};ml.page=1;mlDrawRecords();}));
  document.querySelectorAll(".model-lab-trial").forEach(button=>button.addEventListener("click",()=>{ml.trial=button.dataset.trial;ml.page=1;mlDrawRecords();}));
  document.querySelector("#model-lab-outcome").addEventListener("change",event=>{ml.outcome=event.target.value;ml.page=1;mlDrawRecords();});
  document.querySelector("#model-lab-clear-band").addEventListener("click",()=>{ml.band=null;ml.page=1;mlDrawRecords();});
  for(const [key,offset] of [["prev",-1],["next",1]])document.getElementById("model-lab-"+key).addEventListener("click",()=>{ml.page+=offset;mlDrawRecords();});
  mlDrawRecords();
}
function mlDrawRecords() {
  const data=ml.result.data,trial=data.trials?.find(r=>r.id===ml.trial);
  const beforeThreshold=data.before_threshold??data.threshold;
  let rows=data.records.map((r,i)=>{const before=data.tool==="ensemble"?r.score:r.before,after=data.tool==="ensemble"?r.score:trial?trial.scores[i]:r.after;const was=before>=beforeThreshold,now=after>=data.threshold;return {...r,before,after,was,now,wasCorrect:was===Boolean(r.actual),correct:now===Boolean(r.actual)};});
  rows=rows.filter(r=>ml.outcome==="all"||ml.outcome==="mistakes"&&!r.correct||ml.outcome==="changed"&&r.was!==r.now||ml.outcome==="new-errors"&&r.wasCorrect&&!r.correct||ml.outcome==="fixed-errors"&&!r.wasCorrect&&r.correct);
  if(ml.band){const bins=data[ml.band.version+"_reliability"].bins.length;rows=rows.filter(r=>Math.min(Math.floor(r[ml.band.version]*bins),bins-1)===ml.band.bin);}
  rows.sort((a,b)=>Number(a.correct)-Number(b.correct)||Math.abs(b.after-b.before)-Math.abs(a.after-a.before)||a.source_row-b.source_row);
  const pages=Math.max(1,Math.ceil(rows.length/25));ml.page=Math.min(Math.max(1,ml.page),pages);
  document.querySelector("#model-lab-records").innerHTML=rows.length?axTable(["Source row","Actual label",data.tool==="ensemble"?"Ensemble score":"Original → new score","Decision","Result","Source"],rows.slice((ml.page-1)*25,ml.page*25).map(r=>`<tr><td>${num(r.source_row)}</td><td>${r.actual?"Spam":"Regular email"}</td><td>${data.tool==="ensemble"?pct(r.after):`${pct(r.before)} → ${pct(r.after)}`}</td><td>${r.now?"Flagged":"Passed"}</td><td>${r.correct?"Correct":"Mistake"}</td><td><button class="text-button model-lab-source" data-source="${r.source_row}">Inspect source</button></td></tr>`).join("")):'<p class="no-jobs">No test records match this view.</p>';
  document.querySelector("#model-lab-record-status").textContent=`${data.tool==="stress"?"Selected trial: "+ml.trial+". ":""}${ml.band?`${ml.band.version==="before"?"Original":"Corrected"} confidence band ${ml.band.bin+1}. `:""}Threshold ${mlThreshold(data.threshold)}. Exports include all test records for the selected calculation.`;
  document.querySelector("#model-lab-range").textContent=`${num(rows.length)} matching test records · page ${num(ml.page)} of ${num(pages)}`;
  document.querySelector("#model-lab-prev").disabled=ml.page<=1;document.querySelector("#model-lab-next").disabled=ml.page>=pages;
  document.querySelectorAll(".model-lab-source").forEach(button=>button.addEventListener("click",()=>axRecordDetail(ml.result.query.run,Number(button.dataset.source),data.tool==="ensemble"?data.members[0].model:data.model,ml.result.query.report_sha256)));
}
async function mlExport(event) {
  const result=ml.result,button=event.currentTarget,scope=button.dataset.labExport;if(!result)return;
  const query={...result.query,scope,trial:ml.trial};button.disabled=true;
  try{await fxDownload((scope==="recipe"?"/api/model-lab-recipe?":"/api/model-lab.csv?")+fxQuery(query),scope==="recipe"?`model-${query.tool}-recipe.json`:`model-${query.tool}-${scope}.csv`);}
  catch(error){if(button.isConnected&&result===ml.result)document.querySelector("#model-lab-status").textContent=error.message;}
  finally{if(button.isConnected&&result===ml.result)button.disabled=false;}
}
