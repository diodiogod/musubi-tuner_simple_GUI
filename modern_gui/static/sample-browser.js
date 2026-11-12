/* Sampling navigation is independent of model loading and preview generation. */
(function(root){
  "use strict";
  const byTime=(a,b)=>(Number(a.modified)||0)-(Number(b.modified)||0)||String(a.path).localeCompare(String(b.path));
  const chronological=items=>[...items].sort((a,b)=>(a.source_index||0)-(b.source_index||0)||String(a.sequence_kind).localeCompare(String(b.sequence_kind))||(Number(a.sequence)||0)-(Number(b.sequence)||0)||byTime(a,b));
  const sourceId=item=>item.source_path||item.source_label||"Current run";
  const checkpointKey=item=>JSON.stringify([sourceId(item),item.sequence_kind,item.sequence]);
  const itemKey=item=>item.path||item.url;
  const promptTitle=item=>item.prompt_name&&item.prompt_name.length<=50?item.prompt_name:(item.prompt_index==null?"Preview":`Prompt ${Number(item.prompt_index)+1}`);
  const shortText=(value,length=84)=>{const text=String(value||"").replace(/\s+/g," ").trim();return text.length>length?text.slice(0,length-1)+"…":text};
  function allItems(data){const map=new Map();for(const g of data?.groups||[])for(const item of g.items)map.set(itemKey(item),item);for(const item of data?.ungrouped||[])map.set(itemKey(item),item);return [...map.values()]}
  function orderedGroups(data){const rank=g=>g.items?.[0]?.sequence_kind==="epoch"?0:g.items?.[0]?.sequence_kind==="step"?1:2;return [...(data?.groups||[])].sort((a,b)=>rank(a)-rank(b)||(Number(a.prompt_index)||0)-(Number(b.prompt_index)||0)||String(a.prompt_name||"").localeCompare(String(b.prompt_name||""))||String(a.key).localeCompare(String(b.key)))}
  function matrices(data){
    return ["epoch","step"].map(kind=>{
      const rows=orderedGroups(data).map(group=>({...group,items:chronological(group.items.filter(i=>i.sequence_kind===kind))})).filter(g=>g.items.length);
      const columns=new Map();rows.forEach(row=>row.items.forEach(item=>columns.set(checkpointKey(item),item)));
      const ordered=chronological([...columns.values()]);
      return {kind,columns:ordered.map(item=>({key:checkpointKey(item),item})),rows:rows.map(row=>{
        const cells=new Map();row.items.forEach(item=>{const key=checkpointKey(item);if(!cells.has(key))cells.set(key,[]);cells.get(key).push(item)});return {...row,cells};
      })};
    }).filter(m=>m.rows.length);
  }
  function toggleSelection(paths,path){const next=paths.filter(p=>p!==path);if(next.length!==paths.length)return next;return paths.length<2?[...paths,path]:[paths[0],path]}
  function comparisonContext(data,paths){
    const all=allItems(data),selected=paths.map(path=>all.find(i=>itemKey(i)===path));
    if(selected.some(i=>!i)||selected[0].media_kind!==selected[1].media_kind)return null;
    const map=new Map();for(const item of selected){const group=data.groups.find(g=>g.items.some(i=>itemKey(i)===itemKey(item)));for(const sibling of group?.items||[item])if(sibling.media_kind===item.media_kind)map.set(itemKey(sibling),sibling)}
    return {key:JSON.stringify(paths),items:chronological([...map.values()]),prompt_name:"Selected previews",prompt_text:""};
  }
  let view="prompt",size=158,selected=[],selecting=false,editTarget="b",videoObserver=null;
  const posters=new Map();
  try{const saved=JSON.parse(localStorage.getItem("musubi-sample-browser")||"{}");if(["prompt","epoch","all"].includes(saved.view))view=saved.view;size=Math.max(120,Math.min(230,Number(saved.size)||158))}catch(_){}
  function save(){try{localStorage.setItem("musubi-sample-browser",JSON.stringify({view,size}))}catch(_){}}
  function setMode(mode){state.sampleMode=mode;$("#samples .sample-workspace").dataset.layout=mode;$$("[data-sample-mode]").forEach(b=>{const active=b.dataset.sampleMode===mode;b.classList.toggle("active",active);b.setAttribute("aria-pressed",String(active))})}
  function media(item){
    if(item.media_kind!=="video")return `<img src="${esc(item.url)}" alt="" loading="lazy">`;
    const cached=posters.get(itemKey(item)+":"+item.modified);
    return (cached?`<img src="${cached}" alt="" loading="lazy">`:`<video class="sb-poster-video" data-poster-key="${esc(itemKey(item)+":"+item.modified)}" data-src="${esc(item.url)}" preload="none" muted playsinline aria-hidden="true"></video>`)+`<span class="sb-video-tag">▶ Video</span>`;
  }
  function hydrateVideos(host){
    videoObserver?.disconnect();
    videoObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{if(!entry.isIntersecting)return;const video=entry.target;videoObserver.unobserve(video);
      const capture=()=>{if(!video.videoWidth||!video.videoHeight)return;try{const canvas=document.createElement("canvas"),scale=Math.min(1,360/video.videoWidth);canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);const url=canvas.toDataURL("image/jpeg",.85);posters.set(video.dataset.posterKey,url);if(posters.size>100)posters.delete(posters.keys().next().value);const img=document.createElement("img");img.src=url;img.alt="";video.replaceWith(img)}catch(_){video.style.visibility="visible"}};
      video.addEventListener("loadeddata",()=>{if(video.duration>0){video.addEventListener("seeked",capture,{once:true});video.currentTime=Math.min(.08,video.duration/2)}else capture()},{once:true});video.addEventListener("error",()=>video.style.visibility="visible",{once:true});video.preload="metadata";video.src=video.dataset.src;video.load();
    }),{rootMargin:"100px"});host.querySelectorAll(".sb-poster-video").forEach(v=>videoObserver.observe(v));
  }
  function tile(item,index,inTable=false){
    const label=`${promptTitle(item)} · ${item.sequence_label||"Preview"}${item.seed!=null?` · seed ${item.seed}`:""}`;
    return `<article class="sb-tile"><button class="sb-thumbnail" data-sample-item="${index}" aria-label="Open ${esc(label)}" title="${esc(label)}">${media(item)}<span class="sb-pick-badge" hidden></span></button>${inTable?"":`<div class="sb-tile-caption"><strong>${esc(promptTitle(item))}</strong><small>${esc(item.sequence_label||"Preview")}${item.seed!=null?` · seed ${esc(item.seed)}`:""}</small></div>`}</article>`;
  }
  function renderGallery(){
    setMode("gallery");const data=state.samples;if(!data)return;
    const items=allItems(data),indices=new Map(items.map((item,index)=>[itemKey(item),index])),host=$("#compare-stage");
    selected=selected.filter(path=>indices.has(path));
    const scrolls=new Map([...host.querySelectorAll(".sb-progress-scroll")].map(x=>[x.dataset.kind,x.scrollLeft]));
    host.innerHTML=`<div class="sb-gallery-toolbar"><div class="sb-organize"><label for="sb-grouping">Group by</label><select id="sb-grouping"><option value="prompt">Prompt progression</option><option value="epoch">Epoch / step</option><option value="all">None · newest first</option></select><span class="sb-output-count">${items.length} previews</span></div><div class="sb-gallery-tools"><label class="sb-size" for="sb-size">Size <input id="sb-size" type="range" min="120" max="230" step="1" value="${size}" aria-label="Thumbnail size"></label><button class="quiet" id="sb-select" aria-pressed="${selecting}">${selecting?"Cancel selection":"Select to compare"}</button></div></div><div class="sb-selection-bar" id="sb-selection" ${selecting?"":"hidden"}><span id="sb-selection-help"></span><button class="primary" id="sb-compare" disabled>Compare 2</button></div><div class="sb-gallery-content" style="--sb-size:${size}px"></div>`;
    const content=host.querySelector(".sb-gallery-content");
    if(!items.length)content.innerHTML='<div class="sb-empty"><strong>No previews yet</strong><p>Generated samples will appear here. Choose another source above to browse earlier runs.</p></div>';
    else if(view==="prompt"){
      const blocks=matrices(data);
      content.innerHTML=blocks.map(block=>{
        const multipleSources=state.sampleIncludeSources||new Set(block.columns.map(c=>sourceId(c.item))).size>1;
        return `<section class="sb-progress-block">${blocks.length>1?`<h3>${block.kind==="epoch"?"Epoch previews":"Step previews"}</h3>`:""}<div class="sb-progress-scroll" data-kind="${block.kind}" tabindex="0" aria-label="${block.kind} progression; scroll horizontally for later checkpoints"><table class="sb-progress-table"><thead><tr><th class="sb-prompt-heading" scope="col">Prompt</th>${block.columns.map(c=>`<th scope="col">${multipleSources?`<small>${esc(c.item.source_label)}</small>`:""}${esc(c.item.sequence_label)}</th>`).join("")}</tr></thead><tbody>${block.rows.map(row=>{const first=row.items[0];return `<tr><th class="sb-prompt-heading" scope="row"><strong>${esc(promptTitle(first))}</strong>${row.prompt_text?`<span class="sb-prompt-snippet" title="${esc(shortText(row.prompt_text,200))}">${esc(shortText(row.prompt_text))}</span>`:""}<small>${first.seed!=null?`Seed ${esc(first.seed)}`:"Unknown seed"}${first.width?` · ${first.width}×${first.height}`:""}</small></th>${block.columns.map(c=>`<td>${row.cells.has(c.key)?row.cells.get(c.key).map(item=>tile(item,indices.get(itemKey(item)),true)).join(""):'<span class="sb-missing" aria-label="No sample for this checkpoint">—</span>'}</td>`).join("")}</tr>`}).join("")}</tbody></table></div></section>`;
      }).join("");
      if(data.ungrouped?.length)content.innerHTML+=`<section class="sb-checkpoint-section"><h3>Other previews</h3><div class="sb-card-grid">${data.ungrouped.map(i=>tile(i,indices.get(itemKey(i)))).join("")}</div></section>`;
    }else if(view==="epoch"){
      const sections=new Map();chronological(items).forEach(item=>{const key=item.sequence_kind?checkpointKey(item):"other";if(!sections.has(key))sections.set(key,[]);sections.get(key).push(item)});
      content.innerHTML=[...sections.values()].map(section=>`<section class="sb-checkpoint-section"><h3>${esc(section[0].sequence_label||"Other previews")} <small>${esc(section[0].source_label||"")} · ${section.length} previews</small></h3><div class="sb-card-grid">${section.map(item=>tile(item,indices.get(itemKey(item)))).join("")}</div></section>`).join("");
    }else content.innerHTML=`<div class="sb-card-grid">${[...items].sort((a,b)=>byTime(b,a)).map(item=>tile(item,indices.get(itemKey(item)))).join("")}</div>`;
    host.querySelector("#sb-grouping").value=view;
    host.querySelector("#sb-grouping").onchange=e=>{view=e.target.value;save();renderGallery()};
    host.querySelector("#sb-size").oninput=e=>{size=Number(e.target.value);content.style.setProperty("--sb-size",`${size}px`);save()};
    ["#sb-grouping","#sb-size"].forEach(selector=>host.querySelector(selector).onblur=()=>queueMicrotask(flushSampleUpdates));
    host.querySelector("#sb-select").onclick=()=>{selecting=!selecting;if(!selecting)selected=[];renderGallery()};
    host.querySelectorAll("[data-sample-item]").forEach(button=>button.onclick=()=>{const item=items[Number(button.dataset.sampleItem)];if(selecting){selected=toggleSelection(selected,itemKey(item));paintSelection(items);return}const group=data.groups.find(g=>g.items.some(i=>itemKey(i)===itemKey(item))),sequence=view==="all"?[...items].sort((a,b)=>byTime(b,a)):chronological(group?.items||data.ungrouped||[item]);openSamplePreview(item,sequence,sequence.findIndex(i=>itemKey(i)===itemKey(item)))});
    host.querySelector("#sb-compare").onclick=()=>{const pair=selected.map(path=>items.find(i=>itemKey(i)===path));if(pair.length===2&&pair[0].media_kind===pair[1].media_kind){selecting=false;openSampleComparison(...pair)}};
    host.querySelectorAll(".sb-progress-scroll").forEach(x=>x.scrollLeft=scrolls.get(x.dataset.kind)||0);
    paintSelection(items);hydrateVideos(host);
  }
  function paintSelection(items){
    const host=$("#compare-stage"),pair=selected.map(path=>items.find(i=>itemKey(i)===path)),compatible=pair.length===2&&pair[0].media_kind===pair[1].media_kind;
    host.querySelector(".sb-gallery-content").classList.toggle("is-selecting",selecting);
    host.querySelectorAll("[data-sample-item]").forEach(button=>{const item=items[Number(button.dataset.sampleItem)],role=selected.indexOf(itemKey(item)),badge=button.querySelector(".sb-pick-badge");button.classList.toggle("is-a",selecting&&role===0);button.classList.toggle("is-b",selecting&&role===1);if(selecting)button.setAttribute("aria-pressed",String(role>=0));else button.removeAttribute("aria-pressed");button.setAttribute("aria-label",`${selecting?"Select":"Open"} ${promptTitle(item)} · ${item.sequence_label||"Preview"}`);badge.hidden=!selecting||role<0;badge.textContent=role===0?"A":"B"});
    host.querySelector("#sb-selection-help").textContent=pair.length===0?"Choose reference A, then comparison B.":pair.length===1?`A · ${pair[0].sequence_label||"Preview"} selected. Choose B.`:!compatible?"Choose two images or two videos to compare.":`A · ${pair[0].sequence_label||"Preview"} ↔ B · ${pair[1].sequence_label||"Preview"}`;
    host.querySelector("#sb-compare").disabled=!compatible;
  }
  function renderRail(){
    const host=$("#sample-groups"),groups=state.samples?.groups||[];host.classList.toggle("empty",!groups.length);$("#sample-count").textContent=new Set(groups.map(g=>g.items[0]?.prompt_identity??g.prompt_index??g.key)).size;let lastKind="";
    host.innerHTML=groups.length?groups.map((group,index)=>{const first=group.items[0],kind=first.sequence_kind,heading=kind!==lastKind?`<h3 class="sb-rail-group-title">${kind==="epoch"?"Epoch previews":kind==="step"?"Step previews":"Other previews"}</h3>`:"";lastKind=kind;return `${heading}<button class="sample-series" data-index="${index}" title="${esc(shortText(group.prompt_text,200))}"><strong>${esc(promptTitle(first))}</strong>${group.prompt_text?`<span class="sb-rail-snippet">${esc(shortText(group.prompt_text,64))}</span>`:""}<small>${first.sequence_kind==="step"?"Step":"Epoch"} previews · ${group.items.length}${first.seed!=null?` · seed ${esc(first.seed)}`:""}</small></button>`}).join(""):"No sample series found.";
    host.querySelectorAll("button").forEach(button=>button.onclick=()=>{compareState.manualSelection=null;setMode("compare");renderComparison(Number(button.dataset.index))});
  }
  function renderFilmstrip(items,onPick){
    const host=$("#sample-filmstrip");
    host.innerHTML=items.map((item,index)=>`<button class="sb-film-tile" data-film-index="${index}" aria-label="${esc(item.source_label||"")} ${esc(item.sequence_label||"Preview")}">${media(item)}<span class="sb-film-role"></span><small>${esc(item.sequence_label||"Preview")}</small></button>`).join("");
    host.querySelectorAll("button").forEach(button=>button.onclick=()=>onPick(Number(button.dataset.filmIndex)));
    host.onkeydown=event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();$(event.key==="ArrowLeft"?"#prev-version":"#next-version")?.click()};
    hydrateVideos(host);
  }
  function updateFilmstrip(items){
    $("#compare-choose-a").innerHTML=`<b>A</b><span>Reference<small>${esc(items[compareState.leftIndex]?.sequence_label||"Preview")}</small></span>`;
    $("#compare-choose-b").innerHTML=`<b>B</b><span>Compare<small>${esc(items[compareState.rightIndex]?.sequence_label||"Preview")}</small></span>`;
    ["a","b"].forEach(role=>$("#compare-choose-"+role).setAttribute("aria-pressed",String(editTarget===role)));
    $("#compare-pin-a").textContent=compareState.pinA?"A pinned":"Pin A";$("#compare-pin-a").setAttribute("aria-pressed",String(compareState.pinA));
    $("#sample-filmstrip").querySelectorAll("[data-film-index]").forEach(button=>{const index=Number(button.dataset.filmIndex),a=index===compareState.leftIndex,b=index===compareState.rightIndex;button.classList.toggle("is-a",a);button.classList.toggle("is-b",b);button.querySelector(".sb-film-role").textContent=a&&b?"A+B":a?"A":b?"B":"";button.setAttribute("aria-pressed",String(a||b))});
    const sourceCount=new Set(items.map(sourceId)).size;
    const multiplePrompts=new Set(items.map(item=>item.prompt_identity||item.prompt_index)).size>1;$("#sample-filmstrip").classList.toggle("multiple-prompts",multiplePrompts);
    $("#sample-filmstrip").querySelectorAll(".sb-film-tile").forEach((button,i)=>{button.title=`${items[i].source_label||"Current run"} · ${items[i].name}`;if(multiplePrompts)button.querySelector("small").innerHTML=`<span>${esc(promptTitle(items[i]))}</span>${esc(items[i].sequence_label)}`;else if(sourceCount>1)button.querySelector("small").textContent=`${items[i].source_label} · ${items[i].sequence_label}`});
    const strip=$("#sample-filmstrip"),current=strip.querySelector(`[data-film-index="${compareState.rightIndex}"]`);
    if(current&&(current.offsetLeft<strip.scrollLeft||current.offsetLeft+current.offsetWidth>strip.scrollLeft+strip.clientWidth))strip.scrollLeft=Math.max(0,current.offsetLeft-strip.clientWidth/2+current.offsetWidth/2);
    const a=items[compareState.leftIndex],b=items[compareState.rightIndex],differences=[];
    if(a.seed!==b.seed)differences.push("different seeds");if(a.prompt_identity&&b.prompt_identity&&a.prompt_identity!==b.prompt_identity)differences.push("different prompts");if(a.width&&b.width&&(a.width!==b.width||a.height!==b.height))differences.push("different sizes");
    $("#sample-comparison-note").textContent=differences.length?`Comparing ${differences.join(" · ")}`:"Click a thumbnail to change "+editTarget.toUpperCase()+" · arrows browse B";
  }
  function setEditTarget(role){editTarget=role}
  function getEditTarget(){return editTarget}
  function renderDetails(item){
    const host=$("#sample-preview-details-content");if(!host)return;
    host.innerHTML=`<h3>${esc(promptTitle(item))}</h3>${item.prompt_text?`<p>${esc(item.prompt_text)}</p>`:""}<dl>${[["Checkpoint",item.sequence_label],["Seed",item.seed],["Size",item.width?`${item.width} × ${item.height}`:null],["Source",item.source_label],["Duration",Number.isFinite(item.duration)?`${item.duration.toFixed(2)} s`:null],...Object.entries(item.sample_settings||{}).map(([key,value])=>[({neg:"Negative prompt",frames:"Requested frames",steps:"Sampling steps",guidance:"Guidance"})[key]||key,value]),["Prompt information",item.metadata_source],["File",item.name]].filter(([,v])=>v!==undefined&&v!==null&&v!=="").map(([k,v])=>`<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>`;
  }
  const api={allItems,orderedGroups,chronological,matrices,toggleSelection,comparisonContext,promptTitle,shortText,renderGallery,renderRail,setMode,renderFilmstrip,updateFilmstrip,setEditTarget,getEditTarget,renderDetails};
  if(typeof module!=="undefined")module.exports=api;else root.SampleBrowser=api;
})(typeof window!=="undefined"?window:this);
