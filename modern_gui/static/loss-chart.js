/* Shared live/history charts. UI ideas inspired by Fizgig; implementation is
 * local so historical Musubi running loss never masquerades as raw step loss. */
(function(root){
  "use strict";
  const finite=v=>typeof v==="number"&&Number.isFinite(v);
  const defaults={smooth:.9,raw:true,reported:false,epoch:false,lr:false,log:false,clip:true};
  const colors={raw:"var(--loss-raw,#76a9ee)",smooth:"var(--loss-smooth,#41cda0)",reported:"var(--loss-reported,#d8b66e)",epoch:"var(--loss-epoch,#ec9471)",lr:"var(--loss-lr,#b39af4)"};
  function normalize(data){
    if(Array.isArray(data))data={points:data.map(p=>[p[0],null,null,p[1],null])};
    const points=(data?.points||[]).filter(p=>Array.isArray(p)&&finite(p[0])&&finite(p[3]));
    const hasRaw=points.some(p=>finite(p[2]));
    return {...data,points,hasRaw,hasLR:points.some(p=>finite(p[4])),hasEpoch:hasRaw&&points.some(p=>finite(p[1]))};
  }
  function ema(points,amount){let value;return points.map(p=>{value=value===undefined?p[1]:amount*value+(1-amount)*p[1];return [p[0],value]})}
  function epochMeans(points){
    const groups=new Map();
    points.forEach(p=>{if(!finite(p[1])||!finite(p[2]))return;const g=groups.get(p[1])||{sum:0,n:0,start:p[0],end:p[0]};g.sum+=p[2];g.n++;g.end=p[0];groups.set(p[1],g)});
    return [...groups].map(([epoch,g])=>[(g.start+g.end)/2,g.sum/g.n,epoch]);
  }
  function domain(values,log,clip){
    let v=values.filter(x=>finite(x)&&(!log||x>0)).map(x=>log?Math.log10(x):x).sort((a,b)=>a-b);
    if(!v.length)return [0,1];
    const trim=clip&&v.length>40?Math.floor(v.length*.01):0;
    let lo=v[trim],hi=v[v.length-1-trim],span=Math.max(hi-lo,Math.abs((hi+lo)/2)*.05,log?.02:1e-5);
    const middle=(lo+hi)/2;return [middle-span*.58,middle+span*.58];
  }
  function model(data,settings){
    const d=normalize(data),s={...defaults,...settings};
    const base=d.points.filter(p=>finite(d.hasRaw?p[2]:p[3])).map(p=>[p[0],d.hasRaw?p[2]:p[3]]);
    const series=[{key:"smooth",label:d.hasRaw?"Smoothed loss":"Smoothed running loss",points:ema(base,Math.min(.99,Math.max(0,s.smooth)))}];
    if(s.raw&&d.hasRaw)series.unshift({key:"raw",label:"Raw loss",points:base});
    if(s.reported)series.push({key:"reported",label:"Reported running average",points:d.points.map(p=>[p[0],p[3]])});
    if(s.epoch&&d.hasEpoch)series.push({key:"epoch",label:"Epoch average",points:epochMeans(d.points)});
    const lr=d.points.filter(p=>finite(p[4])&&p[4]>=0).map(p=>[p[0],p[4]]);
    // Clip raw spikes, but keep the complete smoothed trend in view. Clipping
    // quantiles of a sparse legacy curve can otherwise hide half of the run.
    const rawRange=domain(series.flatMap(x=>x.points.map(p=>p[1])),s.log,s.clip);
    const trendRange=domain(series.filter(x=>x.key!=="raw").flatMap(x=>x.points.map(p=>p[1])),s.log,false);
    return {data:d,series,lr:s.lr?lr:[],range:[Math.min(rawRange[0],trendRange[0]),Math.max(rawRange[1],trendRange[1])]};
  }
  function format(value){if(!finite(value))return "—";return value!==0&&(Math.abs(value)<.001||Math.abs(value)>=10000)?value.toExponential(2):Number(value.toPrecision(4)).toString()}
  function spark(data,{compact=false}={}){
    const m=model(data,{smooth:.85,raw:false}),pts=m.series[0].points;
    if(pts.length<2)return `<span class="run-spark-empty">${compact?'—':'No loss curve saved'}</span>`;
    const x0=pts[0][0],span=pts.at(-1)[0]-x0||1,[lo,hi]=m.range;
    const line=pts.map((p,i)=>`${i?"L":"M"}${(4+(p[0]-x0)/span*164).toFixed(1)},${(34-(p[1]-lo)/(hi-lo)*30).toFixed(1)}`).join(" ");
    return `<svg class="run-spark" viewBox="0 0 172 40" role="img" aria-label="Loss trend"><path d="${line}" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="168" cy="${(34-(pts.at(-1)[1]-lo)/(hi-lo)*30).toFixed(1)}" r="2.5" fill="currentColor"/></svg>${compact?"":`<span class="run-spark-value">${format(pts.at(-1)[1])}<small>latest trend</small></span>`}`;
  }
  let uid=0;
  class Chart{
    constructor(host){
      this.host=host;this.id=`loss-plot-${++uid}`;this.settings={...defaults};
      try{Object.assign(this.settings,JSON.parse(localStorage.getItem("musubi-loss-chart-v1")||"{}"))}catch(_){}
      host.classList.add("loss-chart");
      host.innerHTML=`<div class="loss-chart-toolbar"><div><strong>Loss trend</strong><span class="loss-chart-source"></span></div><div class="loss-chart-tools"><label class="loss-smoothing">Smoothing <input aria-label="Loss smoothing" type="range" min="0" max="0.99" step="0.01"><output></output></label><details class="loss-display"><summary>Display options</summary><div>${[
        ["raw","Raw step loss","The loss of each captured update, before smoothing."],
        ["reported","Running average","The trainer’s reported avr_loss; this is already an average."],
        ["epoch","Epoch average","Mean of captured raw losses for each epoch. Missing steps are not estimated."],
        ["lr","Learning rate","Actual logged scheduler learning rate, on its own right-hand scale. Older logs may not contain it."],
        ["log","Log scale","Useful when loss changes by several orders of magnitude. Zero and negative values cannot appear on this scale."],
        ["clip","Clip outliers","Fit the display to the middle 98% of values. This changes only the view, never training or saved losses."]
      ].map(([key,label,tip])=>`<label title="${tip}"><input type="checkbox" data-chart-option="${key}">${label}</label>`).join("")}</div></details></div></div><div class="loss-chart-plot" tabindex="0" role="img" aria-label="Interactive loss chart. Use arrow keys to inspect captured steps."></div><div class="loss-chart-footer"><div class="loss-chart-legend"></div><div class="loss-chart-readout" aria-live="off">Hover or use arrow keys to inspect a step</div></div><p class="loss-chart-note"></p>`;
      this.plot=host.querySelector(".loss-chart-plot");
      this.range=host.querySelector('input[type="range"]');this.range.value=this.settings.smooth;
      this.range.addEventListener("input",()=>{this.settings.smooth=Number(this.range.value);this.change()});
      host.querySelectorAll("[data-chart-option]").forEach(input=>{input.checked=!!this.settings[input.dataset.chartOption];input.addEventListener("change",()=>{this.settings[input.dataset.chartOption]=input.checked;this.change()})});
      this.plot.addEventListener("pointermove",e=>{const box=this.plot.getBoundingClientRect();this.inspect(Math.max(0,Math.min(1,(e.clientX-box.left-58)/(box.width-58-(this.m?.lr.length?72:18)))))});
      this.plot.addEventListener("pointerleave",()=>{this.plot.querySelector(".loss-crosshair")?.setAttribute("visibility","hidden")});
      this.plot.addEventListener("keydown",e=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;e.preventDefault();const n=this.data?.points.length||0;this.selected=e.key==="Home"?0:e.key==="End"?n-1:Math.max(0,Math.min(n-1,(this.selected||0)+(e.key==="ArrowLeft"?-1:1)));this.read(this.selected)});
      this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(host);
      this.update({points:[]});
    }
    change(){try{localStorage.setItem("musubi-loss-chart-v1",JSON.stringify(this.settings))}catch(_){}this.draw()}
    update(data){this.data=normalize(data);this.draw()}
    destroy(){this.observer.disconnect()}
    draw(){
      if(!this.data||!this.host.isConnected)return;
      const s=this.settings,m=this.m=model(this.data,s),d=m.data;
      this.host.querySelector("output").textContent=Number(s.smooth).toFixed(2);
      this.host.querySelector(".loss-chart-source").textContent=d.hasRaw?"Raw step data":"Reported running loss";
      for(const key of ["raw","reported","epoch","lr"]){const input=this.host.querySelector(`[data-chart-option="${key}"]`);input.disabled=key==="epoch"?!d.hasEpoch:key==="lr"?!d.hasLR:key==="raw"?!d.hasRaw:!d.points.length;input.checked=!input.disabled&&!!s[key];input.closest("label").classList.toggle("unavailable",input.disabled)}
      this.host.querySelector(".loss-chart-note").textContent=d.points.length?(d.hasRaw?(d.note||"Lower loss alone does not mean better samples."):"Historical curve: raw step loss wasn’t recorded. Lower loss is not an image-quality score."):"";
      if(d.points.length<2){this.plot.innerHTML='<div class="loss-chart-empty"><span>↗</span><strong>Waiting for a loss curve</strong><small>It appears after a few training updates. Older runs may have no saved curve.</small></div>';this.host.querySelector(".loss-chart-legend").textContent="";return}
      const width=Math.max(280,this.plot.clientWidth),height=260,L=58,R=m.lr.length?72:18,T=18,B=34,W=width-L-R,H=height-T-B;
      const x0=d.points[0][0],x1=d.points.at(-1)[0],xs=x1-x0||1,[lo,hi]=m.range;
      const x=v=>L+(v-x0)/xs*W,y=v=>T+(1-((s.log?Math.log10(v):v)-lo)/(hi-lo))*H;
      this.coords={x,x0,xs};
      const path=(pts,fn)=>{let started=false;return pts.map(p=>{if(!finite(p[1])||(fn===y&&s.log&&p[1]<=0)){started=false;return ""}const out=`${started?"L":"M"}${x(p[0]).toFixed(2)},${fn(p[1]).toFixed(2)}`;started=true;return out}).join(" ")};
      let svg=`<svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><defs><clipPath id="${this.id}"><rect x="${L}" y="${T}" width="${W}" height="${H}"/></clipPath></defs>`;
      for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,py=T+(1-i/4)*H;svg+=`<line class="loss-grid" x1="${L}" x2="${width-R}" y1="${py}" y2="${py}"/><text x="${L-10}" y="${py+4}" text-anchor="end">${format(s.log?10**v:v)}</text>`}
      for(let i=0;i<5;i++){const step=x0+xs*i/4;svg+=`<text x="${x(step)}" y="${height-10}" text-anchor="middle">${Math.round(step).toLocaleString()}</text>`}
      let epoch;const epochs=d.points.filter(p=>{if(!finite(p[1])||p[1]===epoch)return false;epoch=p[1];return true});
      let lastLabel=-Infinity;
      for(const p of epochs){const px=x(p[0]);if(px-lastLabel<65)continue;lastLabel=px;svg+=`<line class="loss-epoch-line" x1="${px}" x2="${px}" y1="${T}" y2="${height-B}"/><text class="loss-epoch-label" x="${px+5}" y="${T+11}">E${p[1]}</text>`}
      svg+=`<g clip-path="url(#${this.id})">`;
      for(const series of m.series)svg+=`<path d="${path(series.points,y)}" class="loss-line loss-line-${series.key}" stroke="${colors[series.key]}"/>`;
      const epochSeries=m.series.find(series=>series.key==="epoch");
      if(epochSeries)for(const p of epochSeries.points){if(!s.log||p[1]>0)svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3" fill="${colors.epoch}"/>`}
      if(m.lr.length){const peak=Math.max(...m.lr.map(p=>p[1]),1e-12)*1.05;svg+=`<path d="${path(m.lr,v=>T+(1-v/peak)*H)}" class="loss-line loss-line-lr" stroke="${colors.lr}"/>`;this.lrPeak=peak}
      svg+=`<line class="loss-crosshair" visibility="hidden" x1="0" x2="0" y1="${T}" y2="${height-B}"/></g>`;
      if(m.lr.length){for(let i=0;i<3;i++)svg+=`<text class="loss-lr-label" x="${width-R+10}" y="${T+H*(1-i/2)+4}">${format(this.lrPeak*i/2)}</text>`}
      svg+="</svg>";this.plot.innerHTML=svg;
      this.host.querySelector(".loss-chart-legend").innerHTML=m.series.map(series=>`<span><i style="--series:${colors[series.key]}"></i>${series.label}</span>`).join("")+(m.lr.length?`<span><i class="dashed" style="--series:${colors.lr}"></i>Learning rate · right axis</span>`:"");
    }
    inspect(fraction){const target=this.coords.x0+fraction*this.coords.xs;let best=0;this.data.points.forEach((p,i)=>{if(Math.abs(p[0]-target)<Math.abs(this.data.points[best][0]-target))best=i});this.read(best)}
    read(index){const p=this.data?.points[index];if(!p||!this.coords)return;this.selected=index;const line=this.plot.querySelector(".loss-crosshair");if(line){line.setAttribute("visibility","visible");line.setAttribute("x1",this.coords.x(p[0]));line.setAttribute("x2",this.coords.x(p[0]))}this.host.querySelector(".loss-chart-readout").textContent=`Step ${p[0].toLocaleString()}${finite(p[1])?` · Epoch ${p[1]}`:""}${finite(p[2])?` · Loss ${format(p[2])}`:""} · Running ${format(p[3])}${finite(p[4])?` · LR ${format(p[4])}`:""}`}
  }
  const api={Chart,normalize,ema,epochMeans,domain,model,spark};
  if(typeof module!=="undefined")module.exports=api;else root.LossCharts=api;
})(typeof window!=="undefined"?window:this);
