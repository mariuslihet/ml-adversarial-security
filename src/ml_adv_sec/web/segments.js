"use strict";

const sx={slices:{model:"clean",feature:0,bands:4,band:-1,outcome:"all",page:1,size:25},shift:{partition:"test",label:"all",feature:0,outside:false,page:1,size:25,sort:"cdf_distance"},slicesSeq:0,shiftSeq:0,slicesResult:null,shiftResult:null,names:[],namesHash:null};
function sxFeatureOptions(selected) {return sx.namesHash===state.reportHash?fxOptions(sx.names.map((name,i)=>[i,featureName(name)]),selected):'<option>Reading measurement names…</option>';}
function sxBand(row) {return row.lower==null?(row.upper==null?"All values":`Below ${measurement(row.upper)}`):row.upper==null?`At least ${measurement(row.lower)}`:`${measurement(row.lower)} to below ${measurement(row.upper)}`;}
function sxShiftBand(row,data) {
  const stats=data.selected.reference;
  if(row.band===0)return `Below training minimum (${measurement(stats.minimum)})`;
  if(row.band===data.histogram.length-1)return `Above training maximum (${measurement(stats.maximum)})`;
  if(stats.minimum===stats.maximum)return `Exactly ${measurement(stats.minimum)}`;
  if(row.band===data.histogram.length-2)return `${measurement(row.lower)} through ${measurement(stats.maximum)} (included)`;
  return sxBand(row);
}
function sxNames(data) {sx.names=data.features;sx.namesHash=state.reportHash;}
function sxDisableExports(kind) {for(const name of ["summary","records"]){const button=document.getElementById(`${kind}-export-${name}`);if(button)button.disabled=true;}}
function sxExportButtons(kind) {return `<div class="row-actions"><button id="${kind}-export-summary" class="button" disabled>${kind==="slices"?"Export band results":"Export all measurements"}</button><button id="${kind}-export-records" class="button primary" disabled>Export matching records</button></div>`;}

function slicesPage() {
  if(sx.slicesHash!==state.reportHash){sx.slices.band=-1;sx.slices.page=1;sx.slicesHash=state.reportHash;}
  if(![...state.report.clean,...state.report.poisoning].some(r=>r.model===sx.slices.model))sx.slices.model="clean";
  const d=sx.slices;
  return `<section class="panel table-panel segment-panel"><div class="panel-head"><div><h2>Where does this filter make mistakes?</h2><p class="panel-description">Break down actual held-out decisions by a message measurement. Boundaries come from training records, and scores are recalculated from the saved model.</p></div>${sxExportButtons("slices")}</div><div class="record-toolbar"><label>Filter<select id="slices-model">${axModelOptions(state.report,d.model)}</select></label><label>Measurement<select id="slices-feature" disabled>${sxFeatureOptions(d.feature)}</select></label><label>Requested value bands<select id="slices-bands">${fxOptions([[4,"4"],[5,"5"],[10,"10"]],d.bands)}</select></label><label>Record results<select id="slices-outcome">${fxOptions([["all","All results"],["missed","Spam missed"],["false-alarm","Regular email flagged"],["correct","Correct decisions"]],d.outcome)}</select></label></div><p id="slices-status" class="view-status" role="status">Reading source measurements and recalculating model scores…</p><div id="slices-body" aria-live="polite"></div><div class="record-actions"><label>Record band<select id="slices-band" disabled><option value="-1">All measured bands</option></select></label><button id="slices-clear" class="button">Show all records</button></div><div id="slices-records"></div><div class="page-controls"><span id="slices-range"></span><div class="row-actions"><label>Rows per page<select id="slices-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],d.size)}</select></label><button id="slices-prev" class="button" disabled>Previous</button><button id="slices-next" class="button" disabled>Next</button></div></div><div class="table-foot">Band counts cover the full test set; record filters affect only the list and record export. Tied training values reduce the number of distinct bands. Empty groups have no rate. This is exploratory analysis of one held-out partition, not a new independent evaluation.</div></section>`;
}

async function sxLoadSlices() {
  const context=fxContext(),request=++sx.slicesSeq,view={...sx.slices};
  const query={run:context.run,report_sha256:context.hash,...view};
  sx.slicesResult=null;sxDisableExports("slices");
  for(const id of ["slices-prev","slices-next","slices-band"])document.getElementById(id).disabled=true;
  document.querySelector("#slices-status").textContent="Checking training boundaries, source rows, and fresh model scores…";
  document.querySelector("#slices-body").innerHTML="";document.querySelector("#slices-records").innerHTML="";document.querySelector("#slices-range").textContent="";
  try {
    const data=await readJSON("/api/error-slices?"+fxQuery(query));
    if(request!==sx.slicesSeq||!fxCurrent(context)||state.page!=="slices")return;
    sx.slices.page=data.page;sx.slicesResult={data,query};sxNames(data);
    document.querySelector("#slices-feature").innerHTML=sxFeatureOptions(data.feature_index);document.querySelector("#slices-feature").disabled=false;
    document.querySelector("#slices-band").innerHTML=fxOptions([[-1,"All measured bands"],...data.bands.map(r=>[r.band,sxBand(r)])],data.band);document.querySelector("#slices-band").disabled=false;
    document.querySelector("#slices-body").innerHTML=`<div class="confusion-grid">${[["Spam caught",data.overall.caught],["Spam missed",data.overall.missed],["Regular email flagged",data.overall.false_alarms],["Regular email passed",data.overall.clear]].map(([name,count])=>`<div><span>${name}</span><strong>${num(count)}</strong><small>All ${num(data.overall.tested)} test records</small></div>`).join("")}</div>${axTable(["Value band","Training records","Test records","Spam caught","Spam missed","False alarms","Error rate","Spam caught rate","False-alarm rate","Records"],data.bands.map(r=>`<tr><td title="${esc(JSON.stringify({lower:r.lower,upper:r.upper}))}" class="method-name">${esc(sxBand(r))}</td><td>${num(r.training_records)}</td><td>${num(r.tested)}</td><td>${num(r.caught)}</td><td>${num(r.missed)}</td><td>${num(r.false_alarms)}</td><td>${pct(r.error_rate)}</td><td>${pct(r.recall)}</td><td>${pct(r.fpr)}</td><td><button class="text-button slices-show-band" data-band="${r.band}">Show records</button></td></tr>`).join(""))}<div class="table-foot">${num(data.bands.length)} distinct bands from ${num(data.requested_bands)} requested. Each lower bound is included; each upper bound is excluded. The first and last bands cover values beyond the training range.</div>`;
    document.querySelector("#slices-records").innerHTML=data.rows.length?axTable(["Source row",esc(featureName(data.feature)),"Actual label","Fresh spam score","Decision","Result","Details"],data.rows.map(r=>`<tr><td>${num(r.source_row)}</td><td>${measurement(r.value)}</td><td>${r.actual?"Spam":"Regular email"}</td><td title="${r.score}">${pct(r.score,4)}</td><td>${axFlag(r.flagged)}</td><td>${r.correct?"Correct":"Wrong"}</td><td><button class="text-button inspect-source" data-source-row="${r.source_row}">Inspect record</button></td></tr>`).join("")):'<p class="no-jobs">No records match this band and result filter.</p>';
    bindRecordDetailButtons(document.querySelector("#slices-records"),context.run,view.model,context.hash);
    document.querySelectorAll(".slices-show-band").forEach(button=>button.addEventListener("click",()=>{sx.slices.band=Number(button.dataset.band);sx.slices.page=1;sxLoadSlices();}));
    document.querySelector("#slices-status").textContent=`Measured ${num(data.overall.tested)} test records. Decision threshold: ${data.threshold>1?"Above 100%":pct(data.threshold,4)}. Largest fresh-score difference: ${data.maximum_score_difference.toExponential(2)}.`;
    document.querySelector("#slices-range").textContent=`${num(data.total)} matching records · page ${num(data.page)} of ${num(data.pages)}`;
    document.querySelector("#slices-prev").disabled=data.page<=1;document.querySelector("#slices-next").disabled=data.page>=data.pages;
    for(const name of ["summary","records"])document.getElementById("slices-export-"+name).disabled=false;
  }catch(error){if(request===sx.slicesSeq&&fxCurrent(context)&&state.page==="slices")document.querySelector("#slices-status").textContent=error.message;}
}

function shiftPage() {
  const d=sx.shift;
  return `<section class="panel table-panel segment-panel"><div class="panel-head"><div><h2>How different are the held-out records?</h2><p class="panel-description">Compare actual measurements with the training partition. This compares dataset splits, not change over time.</p></div>${sxExportButtons("shift")}</div><div class="record-toolbar"><label>Compare training records with<select id="shift-partition">${fxOptions([["test","Held-out test records"],["val","Validation records"]],d.partition)}</select></label><label>Labels on both sides<select id="shift-label">${fxOptions([["all","All email"],["spam","Spam only"],["regular","Regular email only"]],d.label)}</select></label><label>Selected measurement<select id="shift-feature" disabled>${sxFeatureOptions(d.feature)}</select></label><label>Order measurement list<select id="shift-sort">${fxOptions([["cdf_distance","Largest distribution gap"],["outside_count","Most values outside training range"],["scaled_distance","Largest scaled distance"]],d.sort)}</select></label></div><p id="shift-status" class="view-status" role="status">Comparing the original source measurements…</p><div id="shift-summary" aria-live="polite"></div><div id="shift-detail"></div><div class="record-actions"><label class="outside-choice"><input id="shift-outside" type="checkbox" ${d.outside?"checked":""}> Only records outside this measurement's training range</label><label>Rows per page<select id="shift-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],d.size)}</select></label></div><div id="shift-records"></div><div class="page-controls"><span id="shift-range"></span><div class="row-actions"><button id="shift-prev" class="button" disabled>Previous</button><button id="shift-next" class="button" disabled>Next</button></div></div><div class="table-foot">Distribution gap is the largest difference in cumulative shares: zero means the observed distributions match; one means complete separation. Raw distances use each measurement's units. Scaled distance uses the middle-half spread of training values, falling back to training standard deviation; constant training values have no scaled distance. These comparisons do not predict future model performance.</div></section>`;
}

function sxDrawShiftSummary() {
  const result=sx.shiftResult;if(!result)return;
  const rows=[...result.data.rows].sort((a,b)=>(b[sx.shift.sort]??-Infinity)-(a[sx.shift.sort]??-Infinity)||a.column-b.column);
  document.querySelector("#shift-summary").innerHTML=`<div class="segment-scroll">${axTable(["Measurement","Distribution gap","Scaled distance","Outside training range","Zero-share change","Details"],rows.map(r=>`<tr><td class="method-name">${esc(featureName(r.feature))}${r.column===result.data.feature_index?'<span class="method-sub">Selected below</span>':""}</td><td>${pct(r.cdf_distance,3)}</td><td title="${esc(r.scaling||'No training spread')}">${r.scaled_distance==null?"—":measurement(r.scaled_distance)}</td><td>${num(r.outside_count)} · ${pct(r.outside_share)}</td><td>${fxPoints(r.zero_share_change)}</td><td><button class="text-button shift-select-feature" data-feature="${r.column}">Inspect measurement</button></td></tr>`).join(""))}</div>`;
  document.querySelectorAll(".shift-select-feature").forEach(button=>button.addEventListener("click",()=>{sx.shift.feature=Number(button.dataset.feature);sx.shift.page=1;sxLoadShift();}));
}

async function sxLoadShift() {
  const context=fxContext(),request=++sx.shiftSeq,view={...sx.shift};
  const query={run:context.run,report_sha256:context.hash,...view,outside:view.outside?1:0};
  sx.shiftResult=null;sxDisableExports("shift");
  for(const id of ["shift-prev","shift-next"])document.getElementById(id).disabled=true;
  for(const id of ["shift-summary","shift-detail","shift-records"])document.getElementById(id).innerHTML="";
  document.querySelector("#shift-range").textContent="";document.querySelector("#shift-status").textContent="Reading source records and measuring both partitions…";
  try {
    const data=await readJSON("/api/data-shift?"+fxQuery(query));
    if(request!==sx.shiftSeq||!fxCurrent(context)||state.page!=="shift")return;
    sx.shift.page=data.page;sx.shiftResult={data,query};sxNames(data);
    document.querySelector("#shift-feature").innerHTML=sxFeatureOptions(data.feature_index);document.querySelector("#shift-feature").disabled=false;
    sxDrawShiftSummary();
    const selected=data.selected;
    const fields=[["Records","count"],["Lowest value","minimum"],["Highest value","maximum"],["Mean","mean"],["Median","median"],["Standard deviation","std"],["25th percentile","q25"],["75th percentile","q75"]];
    document.querySelector("#shift-detail").innerHTML=`<div class="panel-head analysis-subhead"><div><h2>${esc(featureName(selected.feature))}</h2><p class="panel-description">Distribution distance in source units: ${measurement(selected.raw_distance)}. ${selected.scaling?`Scaled using ${esc(selected.scaling==="training IQR"?"the middle-half training range":selected.scaling)}.`:"Training values are constant; a scaled distance is undefined."}</p></div></div>${axTable(["Measured quantity","Training records","Comparison records"],fields.map(([name,key])=>`<tr><th scope="row">${name}</th><td>${measurement(selected.reference[key])}</td><td>${measurement(selected.comparison[key])}</td></tr>`).join(""))}<div class="panel-head analysis-subhead"><div><h2>Value-band comparison</h2><p class="panel-description">Training-derived boundaries include separate below-range and above-range bands. The final in-range band includes the training maximum.</p></div></div>${axTable(["Value band","Training count","Training share","Comparison count","Comparison share"],data.histogram.map(r=>`<tr><td title="${esc(JSON.stringify({lower:r.lower,upper:r.upper}))}">${esc(sxShiftBand(r,data))}</td><td>${num(r.reference_count)}</td><td>${pct(r.reference_share)}</td><td>${num(r.comparison_count)}</td><td>${pct(r.comparison_share)}</td></tr>`).join(""))}<div class="panel-head analysis-subhead"><div><h2>Source records for this measurement</h2><p class="panel-description">Export includes all 57 original measurements for every matching record, not just this page.</p></div></div>`;
    document.querySelector("#shift-records").innerHTML=data.records.length?axTable(["Source row","Actual label","Original value","Training range"],data.records.map(r=>`<tr><td>${num(r.source_row)}</td><td>${r.actual?"Spam":"Regular email"}</td><td>${measurement(r.value)}</td><td>${r.outside?"Outside":"Within"}</td></tr>`).join("")):'<p class="no-jobs">No records match the selected range filter.</p>';
    document.querySelector("#shift-status").textContent=`${num(data.training_records)} training records compared with ${num(data.comparison_records)} ${data.partition==="test"?"test":"validation"} records using the same label filter. All 57 measurements come from the authenticated source file.`;
    document.querySelector("#shift-range").textContent=`${num(data.total)} matching records · page ${num(data.page)} of ${num(data.pages)}`;
    document.querySelector("#shift-prev").disabled=data.page<=1;document.querySelector("#shift-next").disabled=data.page>=data.pages;
    for(const name of ["summary","records"])document.getElementById("shift-export-"+name).disabled=false;
  }catch(error){if(request===sx.shiftSeq&&fxCurrent(context)&&state.page==="shift")document.querySelector("#shift-status").textContent=error.message;}
}

function bindSegmentContent() {
  bindInvestigationContent();
  ++sx.slicesSeq;++sx.shiftSeq;sx.slicesResult=null;sx.shiftResult=null;
  if(state.page==="slices") {
    for(const key of ["model","feature","bands","band","outcome","size"])document.getElementById("slices-"+key).addEventListener("change",event=>{sx.slices[key]=["feature","bands","band","size"].includes(key)?Number(event.target.value):event.target.value;if(["feature","bands"].includes(key))sx.slices.band=-1;sx.slices.page=1;sxLoadSlices();});
    document.querySelector("#slices-clear").addEventListener("click",()=>{sx.slices.band=-1;sx.slices.outcome="all";sx.slices.page=1;document.querySelector("#slices-outcome").value="all";sxLoadSlices();});
    document.querySelector("#slices-prev").addEventListener("click",()=>{--sx.slices.page;sxLoadSlices();});document.querySelector("#slices-next").addEventListener("click",()=>{++sx.slices.page;sxLoadSlices();});
    for(const scope of ["summary","records"])document.getElementById("slices-export-"+scope).addEventListener("click",event=>sxExport(event,"slices",scope));
    sxLoadSlices();
  }
  if(state.page==="shift") {
    for(const key of ["partition","label","feature","size"])document.getElementById("shift-"+key).addEventListener("change",event=>{sx.shift[key]=["feature","size"].includes(key)?Number(event.target.value):event.target.value;sx.shift.page=1;sxLoadShift();});
    document.querySelector("#shift-outside").addEventListener("change",event=>{sx.shift.outside=event.target.checked;sx.shift.page=1;sxLoadShift();});
    document.querySelector("#shift-sort").addEventListener("change",event=>{sx.shift.sort=event.target.value;sxDrawShiftSummary();});
    document.querySelector("#shift-prev").addEventListener("click",()=>{--sx.shift.page;sxLoadShift();});document.querySelector("#shift-next").addEventListener("click",()=>{++sx.shift.page;sxLoadShift();});
    for(const scope of ["summary","records"])document.getElementById("shift-export-"+scope).addEventListener("click",event=>sxExport(event,"shift",scope));
    sxLoadShift();
  }
}

async function sxExport(event,kind,scope) {
  const button=event.currentTarget,result=sx[kind+"Result"],context=fxContext(),sequence=sx[kind+"Seq"];
  if(!result)return;button.disabled=true;
  const current=()=>button.isConnected&&fxCurrent(context)&&sequence===sx[kind+"Seq"]&&result===sx[kind+"Result"];
  try{await fxDownload((kind==="slices"?"/api/error-slices.csv?":"/api/data-shift.csv?")+fxQuery({...result.query,scope}),`${kind}-${scope}.csv`);}
  catch(error){if(current()){document.querySelector(kind==="slices"?"#slices-status":"#shift-status").textContent=error.message;sx[kind+"Result"]=null;sxDisableExports(kind);}}
  finally{if(current())button.disabled=false;}
}
