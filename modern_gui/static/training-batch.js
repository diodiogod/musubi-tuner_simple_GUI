/* Nominal batch sizes for the guided recipe. Dataset overrides stay authoritative. */
(function(root){
  "use strict";
  function summary(dataset, accumulation){
    if(!dataset)return null;
    const sources=dataset.datasets||[];
    if(!sources.length)return {text:"No active training sources",detail:"Add an image or video source in Datasets."};
    const sizes=[...new Set(sources.map(source=>Number(source.batch_size??dataset.general?.batch_size??1)))].sort((a,b)=>a-b);
    const count=Number(accumulation??1);
    if(sizes.some(size=>!Number.isInteger(size)||size<1))return {text:"Review dataset batch sizes",detail:"Each source needs a positive whole-number batch size."};
    const batches=`Batch ${sizes.length===1?"size":"sizes"}: ${sizes.join(", ")}`;
    if(!Number.isInteger(count)||count<1)return {text:batches,detail:"Enter a positive whole number for gradient accumulation."};
    const effective=sizes.map(size=>size*count);
    const effectiveText=effective.length===1?effective[0]:`${effective[0]}–${effective[effective.length-1]}`;
    return {text:`${batches} · Effective batch: ${effectiveText}`,detail:`${count} ${count===1?"batch":"batches"} per optimizer update on one GPU. Partial buckets can contain fewer examples.${sizes.length>1?" Sources use different batch sizes; mixed updates can fall between these limits.":""}`};
  }
  const api={summary};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  else root.TrainingBatch=api;
})(typeof window!=="undefined"?window:globalThis);
