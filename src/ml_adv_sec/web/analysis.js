"use strict";

const ax={lab:{model:"clean",mode:"saved",threshold:50,target:1},labSeq:0,labResult:null,pair:{right:null,left_model:"clean",right_model:"tree",category:"disagree",page:1,size:25},pairSeq:0,pairResult:null};
const axModelOptions=(report,value)=>fxOptions([...report.clean,...report.poisoning].map(r=>[r.model,fxModelName(r)]),value);
const axFlag=value=>value?"Flagged":"Passed";
const axTable=(head,body)=>`<div class="table-wrap"><table><thead><tr>${head.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;

function decisionLabPage() {
  if(![...state.report.clean,...state.report.poisoning].some(r=>r.model===ax.lab.model))ax.lab.model="clean";
  const l=ax.lab;
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>What changes when the decision line moves?</h2><p class="panel-description">Fresh predictions from the saved model. A preview measures the consequences without changing the saved run.</p></div><button id="lab-export" class="button primary" disabled>Export preview records</button></div><form id="lab-form"><div class="record-toolbar"><label>Filter<select id="lab-model">${axModelOptions(state.report,l.model)}</select></label><label>Decision rule<select id="lab-mode">${fxOptions([["saved","Saved decision"],["manual","Choose a threshold"],["target","Choose a false-alarm target"]],l.mode)}</select></label><label>Spam-score threshold (%)<input id="lab-threshold" type="number" min="0" max="100" step="any" value="${l.threshold}" required></label><label>False-alarm target (%)<input id="lab-target" type="number" min="0" max="25" step="any" value="${l.target}" required></label></div><div class="record-actions"><p class="panel-description">Target mode chooses the threshold using validation records only. Test records measure the outcome.</p><div class="row-actions"><button class="button" type="button" id="lab-reset">Use saved decision</button><button class="button" type="button" id="lab-reject">Preview flag none</button><button class="button primary" type="submit">Run preview</button></div></div></form><p id="lab-status" class="view-status" role="status">Preparing a measured preview…</p><div id="lab-body" aria-live="polite"></div></section>`;
}

function axLabQuery(reject=false) {
  return {run:state.run,report_sha256:state.reportHash,model:ax.lab.model,mode:reject?"manual":ax.lab.mode,threshold:reject?1.000001:ax.lab.threshold/100,target_fpr:ax.lab.target/100};
}

async function axLoadLab(reject=false) {
  const context=fxContext(),request=++ax.labSeq,query=axLabQuery(reject);
  ax.labResult=null;
  document.querySelector("#lab-export").disabled=true;
  document.querySelector("#lab-body").innerHTML='<p class="no-jobs">Running the model on validation and test records…</p>';
  document.querySelector("#lab-status").textContent="Checking evidence and recalculating model scores…";
  try {
    const data=await readJSON("/api/decision-lab?"+fxQuery(query));
    if(request!==ax.labSeq||!fxCurrent(context)||state.page!=="decision")return;
    ax.labResult={data,query};
    const rows=Object.entries(data.metrics).map(([partition,m])=>`<tr><td class="method-name">${partition==="test"?"Held-out test records":"Validation records"}</td><td>${num(m.tested)}</td><td>${num(m.caught)}</td><td>${num(m.missed)}</td><td>${num(m.false_alarms)}</td><td>${num(m.clear)}</td><td>${pct(m.recall)}</td><td>${pct(m.fpr)}</td><td>${pct(m.precision)}</td></tr>`).join("");
    document.querySelector("#lab-body").innerHTML=`<div class="confusion-grid"><div><span>Preview threshold</span><strong>${data.threshold>1?"Flag none":pct(data.threshold,4)}</strong><small>Saved: ${data.saved_threshold>1?"Flag none":pct(data.saved_threshold,4)}</small></div><div><span>Test decisions changed</span><strong>${num(data.changed_count)}</strong><small>Compared with the saved decision</small></div><div><span>Spam missed · saved / preview</span><strong>${num(data.saved_test.missed)} / ${num(data.metrics.test.missed)}</strong><small>Actual held-out records</small></div><div><span>False alarms · saved / preview</span><strong>${num(data.saved_test.false_alarms)} / ${num(data.metrics.test.false_alarms)}</strong><small>Regular email flagged</small></div></div>${axTable(["Partition","Records","Spam caught","Spam missed","False alarms","Regular passed","Spam caught (%)","False alarms (%)","Precision"],rows)}<div class="panel-head analysis-subhead"><div><h2>Where the test scores fall</h2><p class="panel-description">Observed spam share is measured from the source labels. Empty bands have no average. Brier score: ${Number(data.brier).toFixed(6)}.</p></div></div>${axTable(["Score band","Records","Spam","Regular email","Mean score","Observed spam share"],data.histogram.map((r,i)=>`<tr><td>${pct(r.low,0)}–${pct(r.high,0)}${i===9?" (includes 100%)":""}</td><td>${num(r.count)}</td><td>${num(r.spam)}</td><td>${num(r.regular)}</td><td>${pct(r.mean_score)}</td><td>${pct(r.observed_spam)}</td></tr>`).join(""))}<details class="analysis-details"><summary>Validation trade-offs at ${num(data.validation_curve.length)} measured decision boundaries</summary><p>These boundaries come from actual validation scores. The table samples up to 101 boundaries; it is not an exhaustive search display.</p>${axTable(["Threshold","Spam caught","False alarms","Spam caught (%)","False alarms (%)"],data.validation_curve.map(r=>`<tr><td>${r.threshold>1?"Flag none":pct(r.threshold,4)}</td><td>${num(r.caught)}</td><td>${num(r.false_alarms)}</td><td>${pct(r.recall)}</td><td>${pct(r.fpr)}</td></tr>`).join(""))}</details><div class="panel-head analysis-subhead"><div><h2>Records whose decisions changed</h2><p class="panel-description">${num(data.changed_count)} changed. Showing the first ${num(data.changed_rows.length)}; the CSV includes every validation and test record.</p></div></div>${data.changed_rows.length?axTable(["Source row","Actual label","Score","Saved decision","Preview decision","Details"],data.changed_rows.map(r=>`<tr><td>${num(r.source_row)}</td><td>${r.actual?"Spam":"Regular email"}</td><td>${pct(r.score)}</td><td>${axFlag(r.was_flagged)}</td><td>${axFlag(r.now_flagged)}</td><td><button class="text-button inspect-source" data-source-row="${r.source_row}">Inspect record</button></td></tr>`).join("")):'<p class="no-jobs">No test decisions changed.</p>'}<div class="table-foot">This experiment changes the decision rule, not the trained model. Test labels never choose the target-mode threshold. A validation target does not guarantee the same test false-alarm rate.</div>`;
    bindRecordDetailButtons(document.querySelector("#lab-body"),context.run,query.model,context.hash);
    document.querySelector("#lab-status").textContent=`Fresh validation and test predictions checked. Largest difference from saved test scores: ${data.maximum_score_difference.toExponential(2)}. This preview is not saved into the run.`;
    document.querySelector("#lab-export").disabled=false;
  }catch(error){if(request===ax.labSeq&&fxCurrent(context)&&state.page==="decision"){document.querySelector("#lab-status").textContent=error.message;document.querySelector("#lab-body").innerHTML="";}}
}

function disagreementPage() {
  const choices=(fx.available||[]).filter(r=>!r.error);
  if(!choices.some(r=>r.id===ax.pair.right))ax.pair.right=state.run;
  if(![...state.report.clean,...state.report.poisoning].some(r=>r.model===ax.pair.left_model))ax.pair.left_model="clean";
  return `<section class="panel table-panel"><div class="panel-head"><div><h2>Which records do the filters disagree on?</h2><p class="panel-description">The left side uses the selected run. Match actual source records against another filter or run.</p></div><button id="pair-export" class="button primary" disabled>Export matching records</button></div><div class="record-toolbar"><label>Left filter<select id="pair-left">${axModelOptions(state.report,ax.pair.left_model)}</select></label><label>Right run<select id="pair-run">${fxOptions(choices.map(r=>[r.id,r.name]),ax.pair.right)}</select></label><label>Right filter<select id="pair-right" disabled><option>Reading available filters…</option></select></label><label>Records to show<select id="pair-category">${fxOptions([["disagree","Different decisions"],["left-only","Only left is correct"],["right-only","Only right is correct"],["both-wrong","Both are wrong"],["all","All matched records"]],ax.pair.category)}</select></label></div><p id="pair-status" class="view-status" role="status">Matching source records…</p><div id="pair-body" aria-live="polite"></div><div class="page-controls"><span id="pair-range"></span><div class="row-actions"><label>Rows per page<select id="pair-size">${fxOptions([[10,"10"],[25,"25"],[50,"50"],[100,"100"]],ax.pair.size)}</select></label><button id="pair-prev" class="button" disabled>Previous</button><button id="pair-next" class="button" disabled>Next</button></div></div><div class="table-foot">Only source rows present in both test partitions are compared. Records missing from either side are counted separately. Training and validation records are excluded.</div></section>`;
}

async function axLoadPair() {
  const context=fxContext(),request=++ax.pairSeq,view={...ax.pair};
  ax.pairResult=null;
  for(const id of ["pair-export","pair-prev","pair-next","pair-right"])document.getElementById(id).disabled=true;
  document.querySelector("#pair-status").textContent="Verifying both runs and matching source records…";
  document.querySelector("#pair-body").innerHTML="";
  const current=()=>request===ax.pairSeq&&fxCurrent(context)&&state.page==="disagreements";
  try {
    const payload=view.right===context.run?{report:state.report,evidence:{report_sha256:context.hash}}:await readJSON("/api/report/"+encodeURIComponent(view.right));
    if(!current())return;
    if(![...payload.report.clean,...payload.report.poisoning].some(r=>r.model===view.right_model))ax.pair.right_model=view.right_model="clean";
    document.querySelector("#pair-right").innerHTML=axModelOptions(payload.report,view.right_model);
    document.querySelector("#pair-right").disabled=false;
    const query={left:context.run,right:view.right,left_model:view.left_model,right_model:view.right_model,category:view.category,page:view.page,size:view.size,left_sha:context.hash,right_sha:payload.evidence.report_sha256};
    const data=await readJSON("/api/disagreements?"+fxQuery(query));
    if(!current())return;
    ax.pair.page=data.page;ax.pairResult={data,query};
    document.querySelector("#pair-status").textContent=`${num(data.matched)} source records matched. Left-only test records excluded: ${num(data.left_unmatched)}; right-only: ${num(data.right_unmatched)}. Thresholds: ${data.left.threshold>1?"Flag none":pct(data.left.threshold,4)} / ${data.right.threshold>1?"Flag none":pct(data.right.threshold,4)}.`;
    document.querySelector("#pair-body").innerHTML=`<div class="confusion-grid">${[["Both correct",data.counts.both_correct],["Only left correct",data.counts["left-only"]],["Only right correct",data.counts["right-only"]],["Both wrong",data.counts["both-wrong"]]].map(([name,count])=>`<div><span>${name}</span><strong>${num(count)}</strong><small>Across all matched test records</small></div>`).join("")}</div>${data.rows.length?axTable(["Source row","Actual label","Left score","Right score","Left decision","Right decision","Inspect"],data.rows.map(r=>`<tr><td>${num(r.source_row)}</td><td>${r.actual?"Spam":"Regular email"}</td><td title="${r.left_score}">${pct(r.left_score)}</td><td title="${r.right_score}">${pct(r.right_score)}</td><td>${axFlag(r.left_flagged)} · ${r.left_correct?"correct":"wrong"}</td><td>${axFlag(r.right_flagged)} · ${r.right_correct?"correct":"wrong"}</td><td><button class="text-button pair-inspect-left" data-source-row="${r.source_row}">Left record</button> <button class="text-button pair-inspect-right" data-source-row="${r.source_row}">Right record</button></td></tr>`).join("")):'<p class="no-jobs">No matched records in this view.</p>'}`;
    document.querySelector("#pair-range").textContent=`${num(data.total)} matching records · page ${num(data.page)} of ${num(data.pages)}`;
    document.querySelector("#pair-prev").disabled=data.page<=1;
    document.querySelector("#pair-next").disabled=data.page>=data.pages;
    document.querySelector("#pair-export").disabled=false;
    for(const [side,run,model,hash] of [["left",context.run,view.left_model,context.hash],["right",view.right,view.right_model,payload.evidence.report_sha256]])document.querySelectorAll(".pair-inspect-"+side).forEach(button=>button.addEventListener("click",()=>axRecordDetail(run,Number(button.dataset.sourceRow),model,hash)));
  }catch(error){if(current()){document.querySelector("#pair-status").textContent=error.message;document.querySelector("#pair-range").textContent="";}}
}

function bindRecordDetailButtons(container,run,model,hash) {
  container.querySelectorAll(".inspect-source").forEach(button=>button.addEventListener("click",()=>axRecordDetail(run,Number(button.dataset.sourceRow),model,hash)));
}

async function axRecordDetail(run,sourceRow,model,hash) {
  const context=fxContext(),request=++fx.modalSeq;
  fxOpenDialog("Source record "+num(sourceRow));
  try {
    const data=await readJSON("/api/record-detail?"+fxQuery({run,source_row:sourceRow,model,report_sha256:hash}));
    if(request!==fx.modalSeq||!fxCurrent(context)||!document.querySelector("#feature-dialog").open)return;
    document.querySelector("#feature-dialog-body").innerHTML=`<div class="training-counts"><span>Run: ${esc(data.run_name)}</span><span>Source row ${num(data.source_row)}</span><span>Actual label: ${data.actual?"Spam":"Regular email"}</span></div>${axTable(["Filter","Fresh spam score","Decision threshold","Decision","Result"],data.models.map(r=>`<tr><td>${esc(fxModelName(r))}</td><td title="${r.score}">${pct(r.score,4)}</td><td>${r.threshold>1?"Flag none":pct(r.threshold,4)}</td><td>${axFlag(r.flagged)}</td><td>${r.correct?"Correct":"Wrong"}</td></tr>`).join(""))}<div class="panel-head analysis-subhead"><div><h2>All 57 source measurements</h2><p class="panel-description">Original source values and scaled dataset values. The rounded-input filter also applies its saved rounding step. Locked fields cannot be changed by the attacks.</p></div></div>${axTable(["Measurement","Original value","Scaled dataset value","Attack rule"],data.features.map(r=>`<tr><td>${esc(featureName(r.name))}</td><td>${measurement(r.value)}</td><td>${measurement(r.model_input)}</td><td>${r.locked?"Locked":"Can change"}</td></tr>`).join(""))}<div class="table-foot">Every displayed score was recalculated from this run's pinned model weights and checked against its saved test-score vector. The dataset does not contain original message text.</div>`;
  }catch(error){if(request===fx.modalSeq&&fxCurrent(context)&&document.querySelector("#feature-dialog").open)document.querySelector("#feature-dialog-body").innerHTML=`<p class="no-jobs">${esc(error.message)}</p>`;}
}

function bindAnalysisContent() {
  bindResearchContent();
  ++ax.labSeq;++ax.pairSeq;ax.labResult=null;ax.pairResult=null;
  if(state.page==="decision") {
    const form=document.querySelector("#lab-form");
    const sync=()=>{ax.lab.model=document.querySelector("#lab-model").value;ax.lab.mode=document.querySelector("#lab-mode").value;ax.lab.threshold=Number(document.querySelector("#lab-threshold").value);ax.lab.target=Number(document.querySelector("#lab-target").value);document.querySelector("#lab-threshold").disabled=ax.lab.mode!=="manual";document.querySelector("#lab-target").disabled=ax.lab.mode!=="target";};
    form.addEventListener("input",()=>{sync();++ax.labSeq;ax.labResult=null;document.querySelector("#lab-export").disabled=true;document.querySelector("#lab-status").textContent="Settings changed. Run preview to measure these settings.";document.querySelector("#lab-body").innerHTML="";});
    form.addEventListener("submit",event=>{event.preventDefault();sync();if(form.reportValidity())axLoadLab();});
    document.querySelector("#lab-reset").addEventListener("click",()=>{ax.lab.mode="saved";document.querySelector("#lab-mode").value="saved";sync();axLoadLab();});
    document.querySelector("#lab-reject").addEventListener("click",()=>{sync();axLoadLab(true);});
    document.querySelector("#lab-export").addEventListener("click",event=>axExport(event,"lab"));
    sync();axLoadLab();
  }
  if(state.page==="disagreements") {
    for(const [id,key] of [["pair-left","left_model"],["pair-run","right"],["pair-right","right_model"],["pair-category","category"],["pair-size","size"]])document.getElementById(id).addEventListener("change",event=>{ax.pair[key]=key==="size"?Number(event.target.value):event.target.value;ax.pair.page=1;axLoadPair();});
    document.querySelector("#pair-prev").addEventListener("click",()=>{--ax.pair.page;axLoadPair();});
    document.querySelector("#pair-next").addEventListener("click",()=>{++ax.pair.page;axLoadPair();});
    document.querySelector("#pair-export").addEventListener("click",event=>axExport(event,"pair"));
    axLoadPair();
  }
}

async function axExport(event,kind) {
  const button=event.currentTarget,result=ax[kind+"Result"],context=fxContext(),sequence=ax[kind+"Seq"];
  if(!result)return;
  const current=()=>button.isConnected&&fxCurrent(context)&&sequence===ax[kind+"Seq"]&&result===ax[kind+"Result"];
  button.disabled=true;
  try {await fxDownload((kind==="lab"?"/api/decision-lab.csv?":"/api/disagreements.csv?")+fxQuery(result.query),kind==="lab"?"decision-preview.csv":"filter-disagreements.csv");}
  catch(error){if(current()){document.querySelector(kind==="lab"?"#lab-status":"#pair-status").textContent=error.message;ax[kind+"Result"]=null;}}
  finally{if(current())button.disabled=false;}
}
