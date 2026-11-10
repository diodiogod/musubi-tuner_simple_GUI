/* Run review is separate from recipe editing and training continuation. */
const RunInsights = (()=>{
  const charts=new Map(),observers=new Set();let selection=null,dialogChart=null,requestId=0;
  const params=job=>`source=${encodeURIComponent(job._source)}&index=${job._history_index}&key=${job.insights_key}`;
  function inlineCurve(job){
    if(job.kind&&job.kind!=="training"&&job.kind!=="staged_training"&&job.kind!=="face_refinement")return "";
    const points=job.metrics?.chart_history?{points:job.metrics.chart_history}:(job.metrics?.loss_history||job.loss_history||[]);
    return `<button class="run-spark-button" data-insight="chart" aria-label="Open run review" title="Open loss graph, previews, checkpoints and notes">${LossCharts.spark(points,{compact:true})}</button>`;
  }
  function bind(row,job){
    row.querySelectorAll("[data-insight]").forEach(button=>button.addEventListener("click",()=>{row.querySelector(".job-menu").hidden=true;row.querySelector(".more-job").setAttribute("aria-expanded","false");open(job,button.dataset.insight)}));
    const spark=row.querySelector(".run-spark-button");
    if(!spark||!job.insights_key)return;
    // Lazy summaries: visible runs only. Parsing stays server-side and cached.
    const observer=new IntersectionObserver(entries=>{if(!entries.some(e=>e.isIntersecting))return;observer.disconnect();observers.delete(observer);
      const key=`${job.insights_key}:${job.finished_at||job.status}`;
      const task=charts.get(key)||api(`/api/jobs/insights?${params(job)}`).then(p=>p.chart);
      charts.set(key,task);if(charts.size>100)charts.delete(charts.keys().next().value);
      task.then(data=>{if(spark.isConnected)spark.innerHTML=LossCharts.spark(data,{compact:true})}).catch(()=>{charts.delete(key)});
    },{rootMargin:"150px"});observers.add(observer);observer.observe(spark);
  }
  function tab(name){
    const dialog=$("#run-insights-dialog");
    dialog.querySelectorAll("[data-insight-tab]").forEach(b=>{const selected=b.dataset.insightTab===name;b.classList.toggle("active",selected);b.setAttribute("aria-selected",String(selected));b.tabIndex=selected?0:-1});
    dialog.querySelectorAll("[data-insight-panel]").forEach(p=>p.hidden=p.dataset.insightPanel!==name);
    if(name==="chart")dialogChart?.draw();
    if(name==="files"&&!selection.filesLoaded)loadFiles();
  }
  function initialize(){
    const dialog=$("#run-insights-dialog");
    dialog.querySelectorAll("[data-insight-tab]").forEach(b=>{b.onclick=()=>tab(b.dataset.insightTab);b.onkeydown=e=>{if(!["ArrowLeft","ArrowRight"].includes(e.key))return;e.preventDefault();const buttons=[...dialog.querySelectorAll("[data-insight-tab]")];buttons[(buttons.indexOf(b)+(e.key==="ArrowLeft"?2:1))%3].click();buttons.find(x=>x.classList.contains("active")).focus()}});
    $("#run-note-text").addEventListener("input",()=>{$("#run-note-status").textContent=$("#run-note-text").value===selection?.savedNote?"Saved locally":"Unsaved changes"});
    $("#run-note-save").onclick=async()=>{if(!selection)return;const current=selection,text=$("#run-note-text").value,button=$("#run-note-save");await withBusy(button,"Saving…",async()=>{try{await api("/api/jobs/notes",{method:"POST",body:JSON.stringify({source:current.job._source,index:current.job._history_index,key:current.job.insights_key,note:text})});current.savedNote=text;current.job.user_note=text;if(selection===current)$("#run-note-status").textContent=$("#run-note-text").value===text?"Saved locally":"Unsaved changes";renderJobs()}catch(e){toast(e.message,"error")}})};
    const dirty=()=>selection&&$("#run-note-text").value!==selection.savedNote;
    dialog.addEventListener("cancel",e=>{if(dirty()&&!confirm("Close without saving this run note?"))e.preventDefault()});
    dialog.querySelector(".dialog-close").onclick=e=>{e.stopImmediatePropagation();if(!dirty()||confirm("Close without saving this run note?"))dialog.close()};
    dialog.addEventListener("close",()=>{requestId++;dialogChart?.destroy();dialogChart=null;selection=null;dialog.querySelectorAll("video").forEach(v=>v.pause())});
  }
  async function open(job,name="chart"){
    const dialog=$("#run-insights-dialog");if(!dialog.dataset.ready){initialize();dialog.dataset.ready="1"}
    const id=++requestId;selection={job,savedNote:job.user_note||"",filesLoaded:false};
    $("#run-insights-title").textContent=job.name||job.title||job.output_name||"Training run";
    $("#run-insights-subtitle").textContent=`${job.mode||job.settings?.training_mode||job.settings_snapshot?.training_mode||""} · ${job.status||"unknown"} · ${job.started_at?new Date(job.started_at).toLocaleString():""}`;
    $("#run-note-text").value=selection.savedNote;$("#run-note-status").textContent="Saved locally";
    $("#run-insights-files").innerHTML='<div class="run-files-empty">Loading saved outputs…</div>';
    if(!dialog.open)dialog.showModal();dialogChart?.destroy();dialogChart=new LossCharts.Chart($("#run-insights-chart"));tab(name);
    try{const payload=await api(`/api/jobs/insights?${params(job)}`);if(id===requestId)dialogChart?.update(payload.chart)}catch(e){if(id===requestId)$("#run-insights-chart .loss-chart-note").textContent=e.message}
  }
  async function loadFiles(){
    if(!selection)return;const current=selection;current.filesLoaded=true;
    try{const p=await api(`/api/jobs/insights?${params(current.job)}&files=1`);if(selection!==current)return;const files=p.files||{},checkpoints=files.checkpoints||[],samples=files.samples||[];
      $("#run-insights-files").innerHTML=`<div class="run-files-summary"><span><strong>${samples.length}</strong> previews</span><span><strong>${checkpoints.length}</strong> checkpoints & states</span><button class="quiet" id="run-files-open">Open output folder</button></div><p class="run-files-path">${esc(files.root||"No output path was recorded")}</p><p class="run-files-help">${esc(files.scope_note||"")}</p><div class="run-files-layout"><section><h3>Previews</h3><div class="run-preview-grid">${samples.length?samples.map((item,i)=>`<button class="run-preview-tile" data-run-preview="${i}" aria-label="Open ${esc(item.name)}">${item.media_kind==="video"?`<video src="${esc(item.url)}" preload="metadata" muted></video><span class="run-video-badge">▶ Video</span>`:`<img src="${esc(item.url)}" loading="lazy" alt="${esc(item.name)}">`}<span>${esc(item.sequence_label||item.name)}</span></button>`).join(""):'<p class="run-files-empty">No saved previews found for this run.</p>'}</div></section><section><h3>Checkpoints & states</h3><p class="run-files-help">Use Continue as new or Resume exact in history to train again.</p><div class="run-checkpoint-list">${checkpoints.length?checkpoints.map(c=>`<div><span><strong>${esc(c.name)}</strong><small>${esc(c.kind)} · ${new Date(c.modified*1000).toLocaleString()}</small></span><span>${c.size==null?"folder":`${Math.round(c.size/1048576)} MB`}</span></div>`).join(""):'<p class="run-files-empty">No checkpoint files remain in the recorded folder. They may have been moved.</p>'}</div></section></div>`;
      $("#run-files-open").onclick=()=>openJobPath({dataset:{source:current.job._source,index:current.job._history_index}},"output");
      $("#run-insights-files").querySelectorAll("[data-run-preview]").forEach(button=>button.onclick=()=>{const item=samples[Number(button.dataset.runPreview)];const media=$("#run-media-content");media.innerHTML=item.media_kind==="video"?`<video src="${esc(item.url)}" controls autoplay loop></video>`:`<img src="${esc(item.url)}" alt="${esc(item.name)}">`;$("#run-media-name").textContent=item.name;$("#run-media-dialog").showModal()});
      $("#run-media-dialog").onclose=()=>{$("#run-media-content").innerHTML=""};
    }catch(e){if(selection===current)$("#run-insights-files").textContent=e.message;current.filesLoaded=false}
  }
  function reset(){observers.forEach(o=>o.disconnect());observers.clear()}
  return {inlineCurve,bind,open,reset};
})();
