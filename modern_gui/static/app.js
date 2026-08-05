const state = { settings: {}, schema: null, dataset: null, samplingEstimate: null, step: "model", selectedDataset: 0, datasetTab: "media", datasetMedia: null, datasetMediaPage: 1, datasetMediaQuery: "", datasetMediaFilter: "all", datasetInventories: {}, datasetAudit: null, datasetRawDirty: false, datasetCaptionDirty: false, openDatasetMediaIndex: 0, samples: null, sampleMode: "compare", captureNoticeJob: "", dirty: false, datasetDirty: false, datasetFormDirty: false, activeView: "home", faceReferenceFilter: "all", faceReferencePage: 0, jobPage: 0, openPromptIndex: -1, openStageIndex: -1, promptPreview: null, depthGpuSnapshot: null };
let loadedFaceResult = null;
let controlSequence = 0;
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
const sameLocalPath = (left,right) => String(left||"").replaceAll("\\","/").replace(/\/+/g,"/").replace(/\/$/,"").toLowerCase() === String(right||"").replaceAll("\\","/").replace(/\/+/g,"/").replace(/\/$/,"").toLowerCase();

async function api(path, options = {}) {
  const response = await fetch(path, { headers: {"Content-Type":"application/json", ...(options.headers || {})}, ...options });
  const payload = await response.json();
  if (!response.ok || payload.error) {const error=new Error(payload.error || `Request failed: ${response.status}`);error.status=response.status;throw error}
  return payload;
}
function toast(message, tone = "info") {
  const node = $("#toast"); node.textContent = message; node.dataset.tone = tone; node.setAttribute("role", tone === "error" ? "alert" : "status"); node.classList.add("show");
  clearTimeout(node.timer); node.timer = setTimeout(() => node.classList.remove("show"), 3000);
}
async function withBusy(button, busyLabel, task) {
  if (!button || button.dataset.busy === "true") return;
  const original = button.textContent;
  button.dataset.busy = "true"; button.disabled = true; button.setAttribute("aria-busy", "true");
  if (busyLabel) button.textContent = busyLabel;
  try { return await task(); }
  finally { button.dataset.busy = "false"; button.disabled = false; button.removeAttribute("aria-busy"); button.textContent = original; }
}
function updateSaveState() {
  const dirty = state.dirty || state.datasetDirty;
  const global = $("#save-state"), footer = $("#autosave-note");
  const status = state.dirty && state.datasetDirty ? "Recipe + TOML unsaved" : state.datasetDirty ? "TOML unsaved" : state.dirty ? "Recipe unsaved" : "Saved";
  if (global) { global.textContent = status; global.classList.toggle("dirty", dirty); global.title=`${status}. Click to save the current workspace.`; global.setAttribute("aria-label",global.title); }
  if (footer) footer.textContent = state.dirty ? "Recipe has unsaved changes" : "Recipe saved to this workspace";
}
function setDirty(dirty = true) {
  state.dirty = dirty;
  updateSaveState();
}
function setDatasetDirty(dirty = true) {
  state.datasetDirty = dirty;
  const save=$("#save-dataset"),status=$("#dataset-document-state");
  if(save)save.disabled=!state.dataset||(!state.datasetDirty&&!state.datasetFormDirty&&!state.datasetRawDirty);
  if(status){
    status.textContent=!state.dataset?"Not loaded":(state.datasetRawDirty?"TOML needs parsing":state.datasetFormDirty?"Updating draft…":state.datasetDirty?"Unsaved changes":"Saved");
    status.classList.toggle("dirty",!!state.dataset&&(state.datasetDirty||state.datasetFormDirty||state.datasetRawDirty));
  }
  updateSaveState();
}
function confirmWorkspaceReplacement(action) {
  if(!state.dirty&&!state.datasetDirty&&!state.datasetFormDirty&&!state.datasetRawDirty&&!state.datasetCaptionDirty)return true;
  const drafts=[state.dirty?"training recipe":"",state.datasetDirty||state.datasetFormDirty||state.datasetRawDirty?"dataset TOML":"",state.datasetCaptionDirty?"open caption":""].filter(Boolean).join(" and ");
  return confirm(`${action}\n\nYour unsaved ${drafts} changes will be discarded. Choose Cancel to keep editing.`);
}
function discardWorkspaceDrafts() {
  state.dirty=false;state.datasetDirty=false;state.datasetFormDirty=false;state.datasetRawDirty=false;state.datasetCaptionDirty=false;
  if(state.dataset){state.dataset=null;$("#dataset-source").value="";$("#dataset-list").innerHTML="Load a TOML to begin."}
  updateSaveState();
}
function go(view, {historyMode = "push", focusHeading = true} = {}) {
  state.activeView = view;
  $$(".view").forEach(node => node.classList.toggle("active", node.id === view));
  $$(".nav[data-view]").forEach(node => {
    const active = node.dataset.view === view;
    node.classList.toggle("active", active);
    if (active) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current");
  });
  if (view === "jobs") loadJobs().catch(e => toast(e.message));
  if (view === "samples") loadSamples().catch(e => toast(e.message));
  if (view === "datasets") ensureDatasetLoaded();
  const heading = $(`#${view} h1`);
  if (heading) { heading.tabIndex = -1; if(focusHeading)heading.focus({preventScroll:true}); document.title = `${heading.textContent} · Musubi Studio`; }
  const target=view==="home"?`${location.pathname}${location.search}`:`${location.pathname}${location.search}#${view}`;
  if(historyMode!=="none"&&`${location.pathname}${location.search}${location.hash}`!==target)history[historyMode==="replace"?"replaceState":"pushState"](null,"",target);
  window.scrollTo(0, 0);
  if(view==="run")requestAnimationFrame(()=>keepLiveLogAtBottom());
}
function schemaFields(sectionIds) {
  return state.schema.sections.filter(s => sectionIds.includes(s.id)).flatMap(s => s.fields);
}
const HELP = {
  training_mode: "Selects the Musubi model family and changes which files, defaults, and specialized training tools are available.",
  dataset_config: "A Musubi dataset TOML describing image or video directories, captions, resolution buckets, repeats, and cache locations.",
  output_name: "The stable name used for checkpoints, samples, logs, staged artifacts, continuation, and recovery.",
  starting_point_mode: "New starts from the base model. Weights continues from a LoRA as additional work. State performs verified positional recovery.",
  network_dim_low: "LoRA rank controls adapter capacity and file size. Higher is not automatically better, especially for small datasets.",
  network_dim_high: "High-noise LoRA rank controls adapter capacity for Wan's high-noise branch.",
  network_alpha_low: "LoRA alpha scales the effective update. It is commonly set below or equal to rank.",
  network_alpha_high: "High-noise LoRA alpha scales the effective update for Wan's high-noise branch.",
  learning_rate: "Optimizer step size. Large values learn faster but can damage likeness or generalization.",
  blocks_to_swap: "Moves model blocks between GPU and CPU to reduce VRAM use, with a speed cost.",
  mixed_precision: "Controls training compute precision. BF16 is generally preferred on supported modern NVIDIA GPUs.",
  fp8_base: "Loads compatible base-model weights in FP8 to save VRAM. This does not change the saved LoRA precision.",
  minimax_h3_dit_model: "Required: select minimax_h3_fl2va_pruned_int8_convrot.safetensors. This is the supported ~21 GB Comfy FL2VA checkpoint; the full ~66 GB BF16 DiT is not needed.",
  minimax_h3_convrot_bwd_mode: "Choose bf16. It is the tested setting for a 24 GB GPU. The int8 choice is for advanced, unvalidated experiments.",
  recache_latents: "Rebuild the cached image information before training. Enable this for the first run or after changing images, resolution, or VAE.",
  recache_text: "Rebuild the cached caption information before training. Enable this for the first run or after changing captions or the text encoder.",
  sample_every_n_epochs: "Generate scheduled samples after this many epochs. Fractions are allowed: 0.5 means twice per epoch; the GUI converts it to steps using the dataset estimate.",
  minimax_h3_training_preview_mode: "Choose whether scheduled MiniMax training samples are a safe still image or an experimental five-frame video. This does not change each card's standalone Preview setting.",
  sample_every_n_steps: "Generate scheduled samples after this many optimizer steps. The dataset estimate helps you choose a useful cadence.",
  sample_at_first: "Generate the scheduled samples once before the first training step.",
  save_every_n_epochs: "Save an intermediate LoRA checkpoint after this many completed epochs. Keep this at 1 for a checkpoint after every epoch, or leave it blank/0 to disable epoch-based saves.",
  save_every_n_steps: "Save an intermediate LoRA checkpoint after this many optimizer steps. This is useful for short epochs or fine-grained recovery points; leave it blank/0 to disable step-based saves.",
  rename_final_artifacts_to_epoch: "When enabled, the completed LoRA and saved state are renamed from the plain run name to the final epoch suffix, such as run-000002. Disable it when you want the final artifacts to keep the normal run name.",
  timestep_sampling: "For MiniMax H3, leave this on krea2_shift. The GUI selects it automatically; it does not mean that a Krea model is being used.",
  dop_enabled: "Differential Output Preservation adds a class-preservation objective. It costs extra compute and requires correct trigger/class captions.",
  dop_trigger_word: "The exact subject or concept token used in your training captions. DoP uses it to identify what the LoRA is allowed to learn. It must match the token in the captions exactly.",
  dop_class_word: "A plain description of what the subject should remain, such as 'man', 'woman', 'dog', or 'clothed person'. DoP uses this comparison to discourage unrelated changes.",
  dop_loss_weight: "How strongly DoP protects the class behavior. Higher values preserve more but can weaken LoRA learning. Start with 1.0 and compare against a run with DoP disabled.",
  krea2_weight_noise_sigma: "Adds a very small amount of noise to LoRA updates during training. This may reduce overfitting on small datasets. 0 disables it; 0.0125 is the experimental preset value.",
  krea2_weight_noise_mode: "Relative scales the noise to each weight's size and is the recommended mode. Absolute applies the same noise scale everywhere and is mainly for controlled experiments.",
  krea2_weight_noise_bound_norm: "Limits unusually large noise updates so weight noise is less likely to destabilize training. Keep this enabled when using relative weight noise.",
  krea2_depth_anchor_weight: "Controls how strongly training is nudged toward the pose and body structure of each source image. 0 disables depth. Even small values add substantial compute and VRAM use.",
  krea2_depth_anchor_model: "The frozen Depth Anything model used to compare image structure. The default Small model is the intended balance of speed and memory; change it only when testing another compatible model.",
  krea2_depth_anchor_input_size: "Resolution used by the depth model, not the LoRA training resolution. Larger values can preserve finer structure but cost more VRAM and time. 518 is the tested default.",
  krea2_depth_anchor_gradient_weight: "Controls how much of the depth signal is allowed to flow back into LoRA learning. 0.5 is the tested value. This is different from Depth Anchor Weight, which scales the final depth loss.",
  krea2_depth_anchor_grad_checkpoint: "Recomputes part of the depth calculation during backward pass to save VRAM. Keep it enabled on normal GPUs. Disabling it may be faster, but uses more VRAM and can cause an out-of-memory error.",
  krea2_keep_depth_helpers_on_gpu: "Keeps the frozen depth model and its helper tensors in GPU memory between steps. Enable only when you have plenty of free VRAM and want less CPU-to-GPU loading. Leave disabled for safer memory use; it does not improve LoRA quality.",
  krea2_depth_vae_device: "Select where Krea 2 performs the differentiable VAE decode used by depth anchoring. Training GPU is the established default. Secondary sends only the predicted latent to another visible CUDA GPU, decodes it there, and returns pixels and gradients automatically.\n\nKrea uses a lighter 2D image VAE than MiniMax, so an 8 GB helper GPU may be usable, but this is experimental and not guaranteed. Start with a short run and check the startup log to confirm the device mapping.",
};
const LONG_HELP = new Set(["training_mode","starting_point_mode","timestep_sampling","dop_enabled","krea2_generalization_preset","krea2_depth_anchor_gradient_weight","krea2_depth_anchor_grad_checkpoint","krea2_keep_depth_helpers_on_gpu","blocks_to_swap","fp8_base","minimax_h3_dit_model","minimax_h3_convrot_bwd_mode","minimax_h3_training_preview_mode","recache_latents","recache_text","sample_every_n_epochs","sample_every_n_steps","sample_at_first","save_every_n_epochs","save_every_n_steps","rename_final_artifacts_to_epoch"]);
const LONG_HELP_COPY = {
  training_mode: "The model family controls far more than the visible model path. It selects the correct Musubi training script, cache commands, supported precision options, sampling behavior, and mode-specific settings.\n\nChoose the family of the base model you will actually train. Changing it later preserves your other recipe values, but you should review every model path and the Method step again.",
  starting_point_mode: "New LoRA starts from the base model with a fresh adapter. Use this for a new subject, style, or concept.\n\nContinue from LoRA adds more training to existing adapter weights, but starts a fresh optimizer and schedule. Exact recovery restores a verified saved training state so the optimizer, scheduler, epoch, and step position continue together. Do not use exact recovery merely to extend a completed run.",
  timestep_sampling: "This controls which noise levels the model practices during training. You normally do not need to choose it yourself because each training mode selects an appropriate value.\n\nFor MiniMax H3, leave it on krea2_shift. This is the setting used by the successful 24 GB test. Despite the name, it does not load or train a Krea model; MiniMax H3 simply uses the same style of noise schedule. Change it only when following a specific advanced recipe.",
  dop_enabled: "Differential Output Preservation adds a preservation objective beside the normal training loss. It can reduce unwanted changes outside the trained concept, especially for small or narrow datasets.\n\nIt costs additional compute and depends on correct trigger and class captions. Review the DOP weight and words under Regularization before enabling it.",
  krea2_generalization_preset: "This preset applies coordinated adapter weight-noise and depth-anchor values. It is available for Krea 2 and experimental MiniMax H3.\n\nOff sets both strengths to zero. Weight Noise Only applies relative noise at 0.0125 without loading the depth models. Balanced Experimental combines 0.0125 weight noise with a 0.01 depth anchor. Changing the preset updates the visible advanced values immediately. MiniMax H3 depth is VRAM-heavy, so test it with a short run and conservative block swapping.",
  krea2_depth_anchor_gradient_weight: "This controls how strongly gradients from the structural comparison travel back toward the LoRA. It works inside the depth calculation; Depth Anchor Weight separately controls how much the finished depth loss contributes to total training loss.\n\nKeep 0.5 for initial tests. Raising it does not simply produce 'more accurate depth' and may overpower normal identity or appearance learning.",
  krea2_depth_anchor_grad_checkpoint: "Enabled saves VRAM by discarding intermediate depth calculations and recomputing them during backward pass. The tradeoff is extra computation, so each affected step may be slower.\n\nFor a 24 GB training GPU, keep this enabled. Disable it only when you have measured substantial free VRAM and want to test whether retaining the depth graph improves speed. It does not change the intended depth objective or LoRA quality by itself.",
  krea2_keep_depth_helpers_on_gpu: "Enabled keeps the frozen Depth Anything model and its working tensors on the GPU between training steps. This avoids repeated transfers and can make depth-enabled training faster, but permanently consumes additional VRAM.\n\nDisabled moves those helpers away when they are not being used. This is safer on tight GPUs and does not weaken the depth signal or LoRA quality; it can only be slower. On a 24 GB MiniMax run, leave it disabled unless depth helpers are assigned to a separate GPU with enough memory.",
  blocks_to_swap: "Block swapping reduces peak VRAM by moving inactive transformer blocks between GPU and system memory. More swapped blocks generally use less VRAM but increase transfer overhead and slow each step.\n\nStart with the lowest value that fits your GPU. If a run still runs out of memory, increase gradually; if there is comfortable headroom, lower it for speed.",
  fp8_base: "FP8 base loading reduces VRAM used by compatible model weights. The LoRA is still trained and saved using the recipe's selected training precision.\n\nSupport depends on the model family, GPU, and weight format. If startup fails or output quality changes unexpectedly, disable FP8 first and verify a BF16 baseline.",
  minimax_h3_dit_model: "Select minimax_h3_fl2va_pruned_int8_convrot.safetensors from ComfyUI's models/diffusion_models folder. This experimental image-only trainer operates directly on that frozen ~21 GB FL2VA ConvRot INT8 base while training a BF16 LoRA. You do not need to download or reconstruct the ~66 GB full BF16 transformer.\n\nThe checkpoint contract is deliberately strict: Ref2VA, ordinary BF16, GGUF, and other INT8/quantized layouts are rejected instead of being guessed. Text-encoder and VAE files are used only during their separate cache phases.",
  minimax_h3_text_cache_dtype: "Controls only how the completed caption embeddings are stored. The Comfy-style Qwen3-VL tower still performs its encoding calculations in FP32.\n\nUse bfloat16 (recommended) for caches half the size and lower training-time disk/CPU traffic. Float32 is available for controlled fidelity comparisons. After changing this, rebuild only the Caption/Text Cache; the Image/Latent Cache does not need to be rebuilt.",
  minimax_h3_depth_vae_device: "Select where the frozen MiniMax video VAE performs differentiable depth decoding. Secondary uses another CUDA device while the DiT and Depth Anything remain on the training GPU.\n\nThis decoder needs far more than its ~5 GB weights during backward pass. A secondary GPU with at least 16 GB VRAM is recommended. An 8 GB helper GPU is not supported, even with reduced depth resolution. Confirm the logical device mapping in the training log before relying on a multi-GPU setup.",
  minimax_h3_keep_depth_vae_on_device: "Keep the MiniMax video VAE resident on its selected GPU between depth steps. This can reduce transfer overhead on a dedicated secondary GPU with ample VRAM.\n\nIt does not reduce the VAE's peak backward-pass memory and cannot make an 8 GB helper GPU usable. Disable it when VRAM is tight or the helper GPU is shared. This changes speed and idle VRAM use, not LoRA quality.",
  minimax_h3_depth_every_n_steps: "Run the structural depth correction every N optimizer steps. 1 applies depth every step and is strongest but slowest. 2 or 4 substantially reduces the average depth cost and is a practical experimental starting point. Larger values make depth influence the run less frequently.",
  minimax_h3_convrot_bwd_mode: "Choose bf16. It is the tested and recommended option for a 24 GB GPU. This setting only controls temporary calculations while the LoRA learns: the frozen base remains the ~21 GB ConvRot INT8 checkpoint, and the saved LoRA format does not change.\n\nThe int8 option is an advanced experiment. It requires working Triton kernels and has not been validated on this setup, so it should not be used for a normal first run.",
  minimax_h3_training_preview_mode: "One frame (safe) keeps the current low-memory still preview and is the recommended default for frequent sampling.\n\nFive-frame video (experimental) runs native MiniMax video inference inside the sampling pause, then saves a short MP4. It has no training gradients, but it is slower and can temporarily require more VRAM. If it OOMs, training can still be interrupted, so test it with a conservative cadence first. This controls only scheduled in-training samples; each prompt card keeps its own frame count for the standalone Preview button.",
  recache_latents: "This prepares compact training data from every source image using the selected VAE. Enable it for a dataset's first run and whenever images, image resolution, bucketing, or the VAE changes.\n\nFor MiniMax H3, select minimax_h3_video_vae_fp16.safetensors. Do not use a Wan or Krea VAE. Once a compatible cache is current, you can turn this off on later runs to start faster.",
  recache_text: "This prepares caption information using the selected text encoder. Enable it for a dataset's first MiniMax H3 run and whenever captions or the text encoder changes.\n\nFor MiniMax H3, this phase uses qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors. The large text encoder is unloaded before LoRA training begins. Once the caption cache is current, you can turn this off on later runs to start faster.",
  sample_every_n_epochs: "Generate the scheduled comparison prompts after this many completed epochs. Enter 0.5 to sample twice during each epoch. Musubi accepts whole-number epoch values, so the GUI converts a fractional value to an equivalent optimizer-step cadence using the dataset estimate shown in the Training Plan.",
  sample_every_n_steps: "Generate the scheduled comparison prompts after this many optimizer steps. This is useful when you want a precise cadence inside a short epoch; the estimate beside the controls shows the relationship to your dataset.",
  sample_at_first: "Generate one comparison before the first training step. This gives you a baseline to compare with later checkpoints.",
  save_every_n_epochs: "Save an intermediate LoRA checkpoint after this many completed epochs. The default of 1 saves at each epoch boundary. This is independent from preview sampling: a checkpoint is saved even when no preview is scheduled.",
  save_every_n_steps: "Save an intermediate LoRA checkpoint after this many optimizer steps. Use this for fine-grained recovery points inside an epoch. If both epoch and step cadences are enabled, Musubi can save at either cadence.",
  rename_final_artifacts_to_epoch: "The trainer's final LoRA normally keeps the run name, while periodic checkpoints receive epoch or step suffixes. When this option is enabled, the Modern GUI renames the final LoRA and final saved-state folder to the last epoch suffix, such as run-000002. Disable it to leave the final run-name files untouched."
};
function helpFor(field) {
  return HELP[field.key] || `Advanced Musubi setting: ${field.label}. Leave its default value unless a model-specific recipe tells you to change it. Internal option: ${field.key}.`;
}
function openHelp(field) {
  $("#help-title").textContent = field.label; $("#help-copy").textContent = LONG_HELP_COPY[field.key] || helpFor(field); $("#help-key").textContent = field.key;
  $("#help-dialog").showModal();
}

function applyTheme(value, {syncSetting = true, markChanged = false, persist = true} = {}) {
  const theme = String(value || "").toLowerCase() === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  if (persist) localStorage.setItem("musubi-theme", theme);
  const toggle = $("#theme-toggle");
  if (toggle) {
    const next = theme === "light" ? "dark" : "light";
    toggle.textContent = theme === "light" ? "☾" : "☼";
    toggle.title = `Use ${next} theme`;
    toggle.setAttribute("aria-label", `Use ${next} theme`);
  }
  if (syncSetting && state.settings) {
    state.settings.appearance_mode = theme === "light" ? "Light" : "Dark";
    $$('.field[data-key="appearance_mode"] select').forEach(select=>select.value=state.settings.appearance_mode);
  }
  if (markChanged) sync();
  drawLoss([]);
}

function fieldControl(field, {wide = false} = {}) {
  const wrap = document.createElement("div");
  wrap.className = `field${wide || field.type === "textarea" || field.type === "path" ? " wide" : ""}`;
  wrap.dataset.key = field.key; wrap.dataset.modes = (field.modes || []).join("|"); wrap.dataset.search = `${field.label} ${field.key}`.toLowerCase();
  const id = `setting-${field.key.replace(/[^a-z0-9_-]/gi, "-")}-${++controlSequence}`;
  const descriptionId = `${id}-description`;
  const label = document.createElement("label");
  label.classList.add("field-label");
  label.htmlFor = id;
  label.dataset.tip = helpFor(field);
  label.tabIndex = 0;
  label.append(document.createTextNode(field.label));
  const labelRow = document.createElement("div");
  labelRow.className = "field-label-row";
  labelRow.append(label);
  if (LONG_HELP.has(field.key)) {
    const help = document.createElement("button");
    help.type = "button"; help.className = "help"; help.textContent = "?";
    help.dataset.tip = "Open detailed guidance";
    help.setAttribute("aria-label", `Open detailed guidance for ${field.label}`);
    help.addEventListener("click", event => { event.stopPropagation(); openHelp(field); });
    labelRow.append(help);
  }
  let input, customInput=null;
  if (field.type === "boolean") {
    wrap.classList.add("switch-field");
    const text = document.createElement("div"); text.append(labelRow);
    const hint = document.createElement("small"); hint.textContent = state.settings[field.key] ? "Enabled" : "Disabled"; text.append(hint);
    input = document.createElement("input"); input.type = "checkbox"; input.checked = Boolean(state.settings[field.key]);
    input.addEventListener("change", () => hint.textContent = input.checked ? "Enabled" : "Disabled");
    wrap.append(text, input);
  } else {
    wrap.append(labelRow);
    if (field.type === "select") {
      input = document.createElement("select");
      const options = field.options || [], current = String(state.settings[field.key] ?? "");
      options.forEach(value => input.add(new Option(value, value))); input.value = current || options[0] || "";
      if(field.allow_custom){
        input.add(new Option("Custom…","__custom__"));
        customInput=document.createElement("input");customInput.type="text";customInput.placeholder=`Custom ${field.label.toLowerCase()}`;
        if(current&&!options.includes(current)){input.value="__custom__";customInput.value=current}else customInput.hidden=true;
      }
    } else if (field.type === "textarea") {
      input = document.createElement("textarea"); input.value = state.settings[field.key] ?? "";
    } else {
      input = document.createElement("input");
      input.type = looksNumeric(state.settings[field.key]) ? "number" : "text";
      if (input.type === "number") input.step = "any";
      input.value = state.settings[field.key] ?? "";
      if (field.type === "path") input.placeholder = "Paste or choose a local path";
    }
    if(field.type==="path"){
      const pathRow=document.createElement("div");pathRow.className="path-control";pathRow.append(input);
      const browse=document.createElement("button");browse.type="button";browse.className="quiet";browse.textContent="Browse";
      const directoryKeys=new Set(["output_dir","project_root","logging_dir","convert_output_dir","resume_path"]);
      browse.addEventListener("click",async()=>{try{const result=await api("/api/path/select",{method:"POST",body:JSON.stringify({kind:directoryKeys.has(field.key)?"directory":"file",initial:input.value})});if(result.path){input.value=result.path;commit()}}catch(e){toast(e.message,"error")}});pathRow.append(browse);wrap.append(pathRow);
    }else wrap.append(input);
    if(customInput)wrap.append(customInput);
  }
  input.id = id;
  input.disabled = (field.disabled_modes || []).includes(state.settings.training_mode);
  if (input.disabled) input.title = "Fixed by the selected experimental training mode";
  const description = document.createElement("span");
  description.id = descriptionId; description.className = "sr-only"; description.textContent = helpFor(field);
  input.setAttribute("aria-describedby", descriptionId);
  wrap.append(description);
  const commit = () => {
    if(customInput){customInput.hidden=input.value!=="__custom__";if(input.value==="__custom__")customInput.focus()}
    state.settings[field.key] = field.type === "boolean" ? input.checked : input.value==="__custom__" ? customInput.value : input.value;
    if(field.key === "krea2_generalization_preset" && applyGeneralizationPreset(input.value)) return;
    if (field.key === "training_mode") selectMode(input.value);
    if (field.key === "appearance_mode") applyTheme(input.value, {syncSetting:false});
    if (field.key === "dataset_config") $("#dataset-path").value = input.value;
    sync();
  };
  input.addEventListener(field.type === "select" || field.type === "boolean" ? "change" : "input", commit);
  if(customInput){
    customInput.id = `${id}-custom`;
    customInput.setAttribute("aria-label", `Custom ${field.label}`);
    customInput.addEventListener("input",()=>{state.settings[field.key]=customInput.value;sync()});
  }
  return wrap;
}
function looksNumeric(value) { return typeof value === "number" || (typeof value === "string" && value !== "" && /^-?\d+(\.\d+)?$/.test(value)); }
function applyGeneralizationPreset(preset){
  const values={
    "Off (Baseline)":{noise:"0",depth:"0"},
    "Weight Noise Only":{noise:"0.0125",depth:"0"},
    "Balanced Experimental":{noise:"0.0125",depth:"0.01"}
  }[preset];
  if(!values)return false;
  Object.assign(state.settings,{
    krea2_generalization_preset:preset,
    krea2_weight_noise_sigma:values.noise,
    krea2_weight_noise_mode:"relative",
    krea2_depth_anchor_weight:values.depth,
    krea2_depth_anchor_model:"depth-anything/Depth-Anything-V2-Small-hf",
    krea2_depth_anchor_input_size:"518",
    krea2_depth_anchor_gradient_weight:"0.5",
    krea2_depth_anchor_grad_checkpoint:true
  });
  renderGuided();renderAllSettings();sync();
  toast(`${preset} applied: weight noise ${values.noise}, depth ${values.depth}.`);
  return true;
}
function findField(key) { return state.schema.sections.flatMap(s => s.fields).find(f => f.key === key); }
function appendFields(host, keys) {
  host.innerHTML = "";
  keys.map(findField).filter(Boolean).filter(field => !field.modes?.length || field.modes.includes(state.settings.training_mode)).forEach(field => host.append(fieldControl(field)));
}
async function renderMinimaxDepthHardwareNotice() {
  const notice=$("#minimax-depth-hardware-notice");
  if(!notice)return;
  notice.hidden=false;
  notice.className="issue warning";
  notice.textContent="Checking detected GPU memory for experimental MiniMax depth…";
  try {
    const snapshot=state.depthGpuSnapshot||await api("/api/gpu");
    state.depthGpuSnapshot=snapshot;
    if(!snapshot.available||!snapshot.devices?.length){
      notice.textContent="GPU memory could not be detected. For MiniMax depth on a secondary GPU, use at least 16 GB VRAM; 8 GB helpers are unsupported.";
      return;
    }
    const devices=snapshot.devices.map(device=>`${device.name} (${(device.memory_total/1073741824).toFixed(0)} GB)`).join(" · ");
    const hasSmallGpu=snapshot.devices.some(device=>device.memory_total<16*1024**3);
    notice.className=`issue ${hasSmallGpu?"warning":"ok"}`;
    notice.textContent=`Detected: ${devices}. ${hasSmallGpu?"Do not select an 8 GB-class GPU as the MiniMax depth VAE secondary device; use a 16 GB-or-larger helper GPU, or leave depth disabled.":"A 16 GB-or-larger GPU is present, but confirm the training log maps it to the selected secondary device."}`;
  } catch (_) {
    notice.textContent="GPU memory could not be detected. For MiniMax depth on a secondary GPU, use at least 16 GB VRAM; 8 GB helpers are unsupported.";
  }
}
function renderStartingPoint(host) {
  const current=["new","weights","state"].includes(state.settings.starting_point_mode)?state.settings.starting_point_mode:"new";
  const exactRecovery=current==="state"&&state.settings.recovery_mode===true&&state.settings.resume_exact_position===true;
  const choices=[
    {value:"new",title:"New LoRA",copy:"Fresh adapter and optimizer"},
    {value:"weights",title:"Continue from LoRA",copy:"Keep learned weights; restart the schedule"},
    {value:"state",title:exactRecovery?"Resume exact state":"Continue saved state",copy:exactRecovery?"Restore the optimizer, scheduler, and exact position":"Load saved state; add the configured training"}
  ];
  const section=document.createElement("section");section.className="starting-point-guide";
  section.innerHTML=`<div class="decision-heading"><span><strong>How should this run begin?</strong><small>Choose one source explicitly. Inactive paths are cleared so they cannot affect training invisibly.</small></span><button class="help" type="button" data-tip="Open detailed guidance" aria-label="Open detailed guidance for starting point">?</button></div><div class="starting-point-choices">${choices.map(choice=>`<button type="button" data-starting-point="${choice.value}" class="${current===choice.value?"selected":""}" aria-pressed="${current===choice.value}"><i></i><span><strong>${choice.title}</strong><small>${choice.copy}</small></span></button>`).join("")}</div><div class="starting-point-detail"></div>`;
  section.querySelector(".decision-heading .help").addEventListener("click",()=>openHelp(findField("starting_point_mode")||{key:"starting_point_mode",label:"Starting point"}));
  section.querySelectorAll("[data-starting-point]").forEach(button=>button.addEventListener("click",()=>{
    const next=button.dataset.startingPoint;state.settings.starting_point_mode=next;state.settings.resume_exact_position=false;state.settings.recovery_mode=false;
    if(next==="new"){state.settings.network_weights="";state.settings.resume_path=""}
    else if(next==="weights")state.settings.resume_path="";
    else if(next==="state")state.settings.network_weights="";
    renderGuided();renderAllSettings();sync();
  }));
  const detail=section.querySelector(".starting-point-detail");
  if(current==="weights"){
    detail.innerHTML='<div class="continuation-note"><strong>Additive continuation</strong><span>The selected LoRA initializes the adapter. Training begins with a fresh optimizer and schedule.</span></div>';
    const field=findField("network_weights");if(field)detail.append(fieldControl(field,{wide:true}));
  }else if(current==="state"){
    detail.innerHTML=exactRecovery?'<div class="continuation-note exact"><strong>Exact failed-run recovery</strong><span>This restores the saved optimizer, scheduler, random state, and training position. Do not change the starting-point card before launching.</span></div>':'<div class="continuation-note exact"><strong>Additive saved-state continuation</strong><span>This loads compatible saved state but adds the configured training schedule. Exact failed-run recovery is available only from a verified History entry.</span></div>';
    const field=findField("resume_path");if(field)detail.append(fieldControl(field,{wide:true}));
  }else detail.innerHTML='<div class="continuation-note new"><strong>Fresh training</strong><span>No previous adapter or optimizer state will be loaded.</span></div>';
  host.append(section);
}

function renderGuided() {
  $("#mode-choices").innerHTML = state.schema.modes.map(mode => `<button class="choice-card ${mode === state.settings.training_mode ? "selected" : ""}" data-mode="${esc(mode)}"><i></i><span><strong>${esc(mode)}</strong></span></button>`).join("");
  $$("#mode-choices [data-mode]").forEach(button => button.addEventListener("click", () => selectMode(button.dataset.mode)));
  const mode = state.settings.training_mode;
  const modelKeys = mode === "Krea 2"
    ? ["krea2_dit_model","krea2_text_encoder","vae_model","krea2_turbo_dit","krea2_projector_diff"]
    : mode === "MiniMax H3 (Experimental)" ? ["minimax_h3_dit_model","minimax_h3_text_encoder","minimax_h3_tokenizer","minimax_h3_text_cache_dtype","minimax_h3_convrot_bwd_mode","vae_model"]
    : mode?.startsWith("Flux.2") ? ["flux2_dit_model","flux2_text_encoder","vae_model"]
    : ["is_i2v","dit_high_noise","dit_low_noise","t5_model","clip_model","vae_model"];
  appendFields($("#model-fields"), modelKeys);
  appendFields($("#data-fields"), ["dataset_config","project_root","output_dir","output_name"]);
  renderStartingPoint($("#data-fields"));
  const capacityKeys=mode==="Wan 2.2"?["network_dim_low","network_alpha_low","network_dim_high","network_alpha_high"]:["network_dim_low","network_alpha_low"];
  appendFields($("#method-fields"), ["network_type",...capacityKeys,"learning_rate","optimizer_type","lr_scheduler","max_train_epochs","max_train_steps","timestep_sampling","discrete_flow_shift","krea2_generalization_preset"]);
  if(mode==="Krea 2"||mode==="MiniMax H3 (Experimental)"){
    const presetField=$("#method-fields").querySelector('[data-key="krea2_generalization_preset"]');
    if(presetField){
      const action=document.createElement("div");action.className="field-actions";
      const button=document.createElement("button");button.type="button";button.className="quiet";button.textContent="Apply selected preset";
      button.title="Apply this preset's actual weight-noise and depth settings. Use this after loading an older project whose displayed preset may not match its saved values.";
      button.addEventListener("click",()=>applyGeneralizationPreset(presetField.querySelector("select")?.value||state.settings.krea2_generalization_preset));
      action.append(button);presetField.append(action);
    }
  }
  const depthComputeKeys=["minimax_h3_depth_vae_device","minimax_h3_keep_depth_vae_on_device","minimax_h3_depth_every_n_steps"];
  const kreaDepthComputeKeys=["krea2_depth_vae_device"];
  const dopKeys=["dop_enabled","dop_trigger_word","dop_class_word","dop_loss_weight"];
  const regularizationKeys=schemaFields(["regularization"]).map(field=>field.key).filter(key=>!depthComputeKeys.includes(key)&&!kreaDepthComputeKeys.includes(key)&&!dopKeys.includes(key));
  appendFields($("#regularization-fields"),regularizationKeys);
  const supportsDop=["Krea 2","Flux.2 Klein","MiniMax H3 (Experimental)"].includes(mode);
  $("#dop-settings").hidden=!supportsDop;
  appendFields($("#dop-fields"),dopKeys);
  const depthCompute=$("#minimax-depth-compute");depthCompute.hidden=mode!=="MiniMax H3 (Experimental)";
  appendFields($("#minimax-depth-fields"),depthComputeKeys);
  if(mode==="MiniMax H3 (Experimental)")renderMinimaxDepthHardwareNotice();
  const kreaDepthCompute=$("#krea-depth-compute");kreaDepthCompute.hidden=mode!=="Krea 2";
  appendFields($("#krea-depth-fields"),kreaDepthComputeKeys);
  appendFields($("#performance-fields"), ["mixed_precision","attention_mechanism","gradient_checkpointing","blocks_to_swap","fp8_base","fp8_scaled","persistent_data_loader_workers","max_data_loader_n_workers","compile"]);
  renderReview();
  renderPlan();
  renderFaceWorkspace();
  renderTools();
}
function selectMode(mode) {
  state.settings.training_mode = mode;
  if(mode === "MiniMax H3 (Experimental)"){
    Object.assign(state.settings, {
      network_type:"LoRA", network_dim_low:"16", network_alpha_low:"16",
      blocks_to_swap:"30", mixed_precision:"bf16", attention_mechanism:"sdpa",
      gradient_checkpointing:true, timestep_sampling:"krea2_shift",
      compile:false, fp8_base:false, fp8_scaled:false,
      minimax_h3_tokenizer:state.settings.minimax_h3_tokenizer||"Qwen/Qwen3-VL-32B-Instruct",
      minimax_h3_convrot_bwd_mode:"bf16",
      minimax_h3_depth_vae_device:state.settings.minimax_h3_depth_vae_device||"training",
      minimax_h3_keep_depth_vae_on_device:state.settings.minimax_h3_keep_depth_vae_on_device??false,
      minimax_h3_depth_every_n_steps:state.settings.minimax_h3_depth_every_n_steps||"1",
    });
    const face=faceConfig(),faceBlocks=Number(face.blocks_to_swap);
    face.cfg_scale=1;
    if(!Number.isInteger(faceBlocks)||faceBlocks<30||faceBlocks>48)face.blocks_to_swap=35;
  }
  renderGuided(); renderAllSettings(); sync();
}
function setStep(step) {
  state.step = step;
  $$(".recipe-step").forEach(node => node.classList.toggle("active", node.dataset.step === step));
  $$(".step-pane").forEach(node => node.classList.toggle("active", node.dataset.pane === step));
  const order = ["model","data","method","performance","review"], index = order.indexOf(step);
  $("#step-back").style.visibility = index ? "visible" : "hidden";
  $("#step-next").style.visibility = step === "review" ? "hidden" : "visible";
}
function moveStep(delta) {
  const order = ["model","data","method","performance","review"];
  setStep(order[Math.max(0, Math.min(order.length - 1, order.indexOf(state.step) + delta))]);
}
function renderReview() {
  const cacheKeys=state.settings.use_staged_training
    ? [["staged_recache_latents","Rebuild image cache between stages"],["staged_recache_text","Rebuild text cache between stages"]]
    : [["recache_latents","Rebuild image cache"],["recache_text","Rebuild text cache"]];
  const cachePreparation=cacheKeys.filter(([key])=>state.settings[key]).map(([,label])=>label).join(" + ")||"Reuse existing caches";
  const rows = [
    ["Model", state.settings.training_mode || "Not selected"],
    ["Dataset", state.settings.dataset_config || "Not selected"],
    ["Output", [state.settings.output_dir, state.settings.output_name].filter(Boolean).join("\\") || "Not configured"],
    ["Starting point", state.settings.starting_point_mode==="state"?"Exact saved-state recovery":state.settings.starting_point_mode==="weights"?"Continue from existing LoRA":"New LoRA"],
    ["Method", `${state.settings.network_type || "LoRA"} · rank ${state.settings.network_dim_low || "—"}${state.settings.training_mode==="Wan 2.2"&&state.settings.network_dim_high?` / ${state.settings.network_dim_high}`:""}`],
    ["Schedule", state.settings.max_train_steps ? `${state.settings.max_train_steps} steps` : `${state.settings.max_train_epochs || "—"} epochs`],
    ["Precision", state.settings.mixed_precision || "Default"],
    ["Cache preparation", cachePreparation],
  ];
  $("#review-summary").innerHTML = rows.map(([label,value]) => `<div class="review-row"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
}
function renderHome() {
  const mode = state.settings.training_mode, dataset = state.settings.dataset_config, output = state.settings.output_name;
  $("#workspace-name").textContent = output || mode || "No training selected";
  [["#ready-model",mode,mode || "Select a family above"],["#ready-data",dataset,dataset || "No TOML selected"],["#ready-output",output,output || "No output configured"]].forEach(([selector,value,copy]) => {
    const node = $(selector); node.classList.toggle("ready", Boolean(value)); node.querySelector("small").textContent = copy;
  });
  $("#step-model-summary").textContent = mode || "Choose a family";
  $("#step-data-summary").textContent = dataset ? dataset.split(/[\\/]/).pop() : "Connect training data";
}
function sync(markChanged = true) {
  $("#settings-json").value = JSON.stringify(state.settings, null, 2);
  renderHome(); renderReview(); renderTrainingSummary();
  if (markChanged) setDirty(true);
}

function objectField(label,key,value,onChange,{type="text",options=null,wide=false,help="",browseKind=""}={}) {
  const wrap=document.createElement("div");wrap.className=`field${wide?" wide":""}`;
  const id=`object-${String(key).replace(/[^a-z0-9_-]/gi,"-")}-${++controlSequence}`,descriptionId=`${id}-description`;
  const labelNode=document.createElement("label");labelNode.className="field-label";labelNode.htmlFor=id;labelNode.dataset.tip=help||`${label} is saved with this workflow.`;labelNode.tabIndex=0;labelNode.textContent=label;wrap.append(labelNode);
  let input;
  if(type==="boolean"){wrap.classList.add("switch-field");input=document.createElement("input");input.type="checkbox";input.checked=Boolean(value)}
  else if(type==="select"){input=document.createElement("select");(options||[]).forEach(x=>input.add(new Option(x.label||x,x.value??x)));input.value=value??"";wrap.append(input)}
  else if(type==="textarea"){input=document.createElement("textarea");input.value=value??"";wrap.append(input)}
  else{input=document.createElement("input");input.type=type;input.value=value??"";wrap.append(input)}
  input.id=id;input.dataset.objectKey=key;
  const description=document.createElement("span");description.id=descriptionId;description.className="sr-only";description.textContent=labelNode.dataset.tip;input.setAttribute("aria-describedby",descriptionId);wrap.append(description);
  const commit=()=>{let next=type==="boolean"?input.checked:input.value;if(type==="number"&&next!=="")next=Number(next);onChange(next);sync()};
  input.addEventListener(type==="select"||type==="boolean"?"change":"input",commit);
  if(browseKind){
    const row=document.createElement("div");row.className="path-control";input.replaceWith(row);row.append(input);
    const browse=document.createElement("button");browse.type="button";browse.className="quiet";browse.textContent="Browse";
    browse.addEventListener("click",()=>withBusy(browse,"Choosing…",async()=>{const result=await api("/api/path/select",{method:"POST",body:JSON.stringify({kind:browseKind,initial:input.value})});if(result.path){input.value=result.path;commit()}}).catch(e=>toast(e.message,"error")));
    row.append(browse);
  }
  if(type==="boolean")wrap.append(input);return wrap;
}
function planStageArtifactLabel(stage,index){
  let label=String(stage?.label||"").trim()||`stage-${index+1}`;
  if(/^\d+$/.test(label))label+=`px`;
  return label.replace(/[<>:"/\\|?*\u0000-\u001f]+/g,"-").replace(/^[ .-]+|[ .-]+$/g,"")||`stage-${index+1}`;
}
function uniqueStageLabel(base="Stage"){
  const existing=new Set((state.settings.staged_training_config||[]).map((stage,index)=>planStageArtifactLabel(stage,index).toLowerCase()));
  let count=1,candidate=String(base||"Stage").trim()||"Stage";
  while(existing.has(planStageArtifactLabel({label:candidate},0).toLowerCase())){
    count+=1;candidate=`${String(base||"Stage").replace(/\s+copy(?:\s+\d+)?$/i,"").trim()||"Stage"} copy${count===2?"":` ${count}`}`;
  }
  return candidate;
}
function defaultPromptForMode(){
  const mode=state.settings.training_mode||"Wan 2.2",turbo=mode==="Krea 2"&&String(state.settings.krea2_turbo_dit||"").trim();
  if(mode==="Krea 2")return {enabled:true,prompt:"",neg:"",width:1024,height:1024,steps:turbo?8:28,seed:42,guidance:turbo?1:5.5,...(turbo?{mu:1.15}:{})};
  if(mode==="MiniMax H3 (Experimental)")return {enabled:true,prompt:"",neg:"",width:768,height:768,frames:39,fps:24,steps:20,seed:42,guidance:1,cfg_scale:1,flow_shift:12};
  if(mode==="Wan 2.2")return {enabled:true,prompt:"",neg:"",width:832,height:480,frames:25,steps:20,seed:42,guidance:5,cfg_scale:1};
  return {enabled:true,prompt:"",neg:"",width:1024,height:1024,steps:20,seed:42,guidance:5,cfg_scale:1};
}
function ensurePreviewSettings(){
  if(typeof state.settings.preview_use_lora!=="boolean")state.settings.preview_use_lora=true;
  if(state.settings.preview_lora_multiplier==null||state.settings.preview_lora_multiplier==="")state.settings.preview_lora_multiplier="1.0";
}
function promptPreviewUsesLora(prompt){ensurePreviewSettings();return typeof prompt?._preview_use_lora==="boolean"?prompt._preview_use_lora:state.settings.preview_use_lora}
function applyPromptModelDefaults(index){
  const prompt=state.settings.sample_prompts_data?.[index];if(!prompt)return;
  const mode=state.settings.training_mode||"current model",preserved=Object.fromEntries(Object.entries(prompt).filter(([key])=>key==="prompt"||key==="enabled"||key.startsWith("_library_")||key.startsWith("_preview_")));
  Object.keys(prompt).forEach(key=>delete prompt[key]);Object.assign(prompt,defaultPromptForMode(),preserved);
  renderPlan();sync();toast(`${mode} defaults applied to prompt ${index+1}.`);
}
function promptCardIssues(prompt,index,{includeDisabled=false}={}){
  if(prompt.enabled===false&&!includeDisabled)return [];
  const issues=[],prefix=`Prompt ${index+1}`;
  if(!String(prompt.prompt||"").trim())issues.push(`${prefix} needs text`);
  [["width","width"],["height","height"],["steps","steps"],["frames","frame count"]].forEach(([key,label])=>{
    const value=String(prompt[key]??"").trim();
    if(value&&(!/^\d+$/.test(value)||Number(value)<1))issues.push(`${prefix} has invalid ${label}`);
  });
  const seed=String(prompt.seed??"").trim();
  if(seed&&!/^-?\d+$/.test(seed))issues.push(`${prefix} seed must be a whole number`);
  [["guidance","guidance"],["cfg_scale","CFG scale"],["flow_shift","flow shift"],["mu","Mu"],["y1","Y1"],["y2","Y2"]].forEach(([key,label])=>{
    const value=String(prompt[key]??"").trim();
    if(value&&!Number.isFinite(Number(value)))issues.push(`${prefix} has invalid ${label}`);
  });
  if(state.settings.training_mode==="MiniMax H3 (Experimental)"){
    if(String(prompt.neg||"").trim())issues.push(`${prefix} cannot use a negative prompt with MiniMax H3`);
    if(Number(prompt.guidance??1)!==1||Number(prompt.cfg_scale??1)!==1)issues.push(`${prefix} must keep MiniMax H3 guidance and CFG at 1.0`);
    const frames=Number(prompt.frames??39);
    if(frames!==1&&(frames<5||(frames-5)%17!==0))issues.push(`${prefix} frames must be 5, 22, 39, ... (or 1 for the legacy still preview)`);
    if(Number(prompt.width||0)%32||Number(prompt.height||0)%32)issues.push(`${prefix} MiniMax H3 size must be a multiple of 32`);
  }
  return issues;
}
function stageCardIssues(stage,index){
  if(!state.settings.use_staged_training||stage.enabled===false)return [];
  const issues=[],raw=String(stage.label||"").trim(),name=raw||`Stage ${index+1}`;
  if(!raw)issues.push(`Stage ${index+1} needs a label`);
  if(/[<>:"/\\|?*\u0000-\u001f]/.test(raw)||/[. ]$/.test(raw))issues.push(`${name} has an unsafe label`);
  const normalized=planStageArtifactLabel(stage,index).toLowerCase(),duplicate=(state.settings.staged_training_config||[]).some((item,itemIndex)=>itemIndex!==index&&item.enabled!==false&&planStageArtifactLabel(item,itemIndex).toLowerCase()===normalized);
  if(duplicate)issues.push(`${name} duplicates another stage label`);
  const limit=String(stage.steps||stage.epochs||"").trim();
  if(!/^\d+$/.test(limit)||Number(limit)<1)issues.push(`${name} needs a positive limit`);
  if(stage.type!=="face_refinement"&&!String(stage.dataset_config||"").trim())issues.push(`${name} needs a dataset TOML`);
  if(stage.type!=="face_refinement"&&stage.dop_mode==="enable"){
    if(!["Krea 2","Flux.2 Klein","MiniMax H3 (Experimental)"].includes(state.settings.training_mode))issues.push(`${name} cannot enable DOP for this model`);
    const strength=Number(stage.dop_loss_weight||state.settings.dop_loss_weight||0),trigger=String(stage.dop_trigger_word||state.settings.dop_trigger_word||"").trim(),classWord=String(stage.dop_class_word||state.settings.dop_class_word||"").trim();
    if(!Number.isFinite(strength)||strength<=0)issues.push(`${name} needs a positive DOP strength`);
    if(!trigger||!classWord||trigger.toLowerCase()===classWord.toLowerCase())issues.push(`${name} needs distinct DOP trigger and class words`);
  }
  if(stage.type==="face_refinement"&&index===(state.settings.staged_training_config||[]).findIndex(item=>item.enabled!==false)){
    const config=state.settings.face_refinement_config||{};
    if(config.input_mode!=="existing_lora"||!String(config.input_lora||"").trim())issues.push(`${name} needs an existing LoRA because it is first`);
  }
  return issues;
}
function promptPlanIssues(){
  const issues=[];
  (state.settings.sample_prompts_data||[]).forEach((prompt,index)=>issues.push(...promptCardIssues(prompt,index)));
  (state.settings.staged_training_config||[]).forEach((stage,index)=>issues.push(...stageCardIssues(stage,index)));
  if(state.settings.use_staged_training&&!(state.settings.staged_training_config||[]).some(stage=>stage.enabled!==false))issues.push("Staged training needs at least one included stage");
  [["sample_every_n_epochs","epoch cadence",/^\d+(?:\.\d+)?$/],["sample_every_n_steps","step cadence",/^\d+$/]].forEach(([key,label,pattern])=>{const value=String(state.settings[key]??"").trim();if(value&&value!=="0"&&(!pattern.test(value)||Number(value)<=0))issues.push(`Sample ${label} must be a positive ${key==="sample_every_n_epochs"?"number":"whole number"} or 0`)});
  [["save_every_n_epochs","epoch checkpoint cadence"],["save_every_n_steps","step checkpoint cadence"]].forEach(([key,label])=>{const value=String(state.settings[key]??"").trim();if(value&&value!=="0"&&(!/^\d+$/.test(value)||Number(value)<1))issues.push(`Save ${label} must be a positive whole number or 0`)});
  return issues;
}
function sampleScheduleLabel(){
  const parts=[];
  if(state.settings.sample_at_first)parts.push("At start");
  const epochs=String(state.settings.sample_every_n_epochs||"").trim(),steps=String(state.settings.sample_every_n_steps||"").trim();
  if(epochs&&epochs!=="0")parts.push(`Every ${epochs} ${Number(epochs)>1?"epochs":"epoch"}`);
  if(steps&&steps!=="0")parts.push(`Every ${steps} steps`);
  return parts.join(" + ")||"Off";
}
function checkpointScheduleLabel(){
  const parts=[],epochs=String(state.settings.save_every_n_epochs||"").trim(),steps=String(state.settings.save_every_n_steps||"").trim();
  if(epochs&&epochs!=="0")parts.push(`Every ${epochs} ${Number(epochs)===1?"epoch":"epochs"}`);
  if(steps&&steps!=="0")parts.push(`Every ${steps} steps`);
  return parts.join(" + ")||"Off";
}
function renderPlanOverview(){
  const prompts=state.settings.sample_prompts_data||[],included=prompts.filter(prompt=>prompt.enabled!==false),invalidIncluded=included.filter(prompt=>promptCardIssues(prompt,prompts.indexOf(prompt)).length),stages=state.settings.staged_training_config||[],activeStages=stages.filter(stage=>stage.enabled!==false),issues=promptPlanIssues();
  const h3=state.settings.training_mode==="MiniMax H3 (Experimental)";
  $("#plan-prompt-count").textContent=`${included.length} / ${prompts.length}`;
  $("#plan-prompt-health").textContent=!prompts.length?"Add a prompt":included.length?`${included.length} used for comparisons`:"All prompts are off";
  const videoCards=h3?included.filter(prompt=>Number(prompt.frames??1)>1).length:0;
  const fiveFrameScheduled=h3&&["five_frame","Five-frame video (experimental)"].includes(state.settings.minimax_h3_training_preview_mode);
  const h3Schedule=h3?(fiveFrameScheduled?" · scheduled 5-frame video (experimental)":videoCards?` · scheduled 1-frame still (${videoCards} video card${videoCards===1?"":"s"})`:" · scheduled 1-frame still"):"";
  $("#plan-sample-schedule").textContent=`${sampleScheduleLabel()}${h3Schedule}`;
  $("#plan-checkpoint-schedule").textContent=checkpointScheduleLabel();
  $("#plan-stage-count").textContent=state.settings.use_staged_training?`${activeStages.length} active`:"Normal run";
  $("#plan-stage-health").textContent=state.settings.use_staged_training?(activeStages.length?"Ordered handoff plan":"Enable at least one stage"):"Uses the main recipe";
  $("#plan-health").textContent=issues.length?`${issues.length} to review`:"Ready";
  $("#plan-health-copy").textContent=issues[0]||"No plan issues";
  $("#plan-overview").classList.toggle("has-issues",issues.length>0);
  $("#prompt-tab-count").textContent=String(prompts.length);$("#stage-tab-count").textContent=String(stages.length);
  $("#plan-save-note").textContent=state.dirty?"Plan has unsaved changes":"Changes stay in this workspace";
  const preview=$("#preview-prompts"),previewMode=["Krea 2","MiniMax H3 (Experimental)"].includes(state.settings.training_mode),h3SelectionInvalid=h3&&included.length!==1,sourceKinds=new Set(included.map(prompt=>promptPreviewUsesLora(prompt))),mixedSources=sourceKinds.size>1,sourceLabel=sourceKinds.values().next().value?"LoRA":"base";
  preview.textContent=h3?(included.length===1?`Generate selected with ${sourceLabel}`:included.length?"Generate individually below":"Enable one prompt to generate"):included.length?`Generate ${included.length} with ${mixedSources?"one source":sourceLabel}`:"Generate enabled";
  preview.disabled=!included.length||Boolean(invalidIncluded.length)||!previewMode||h3SelectionInvalid||mixedSources;
  preview.title=!previewMode?"Standalone prompt preview is available for Krea 2 and experimental MiniMax H3.":!included.length?"Enable one prompt before generating a preview.":invalidIncluded.length?"Fix the included prompt card marked Needs attention before generating its preview.":h3&&included.length>1?"MiniMax H3 safely loads its large models once per prompt. Generate from an individual card, or enable only one prompt.":mixedSources?"Enabled cards mix Base and LoRA. Generate them individually, or set them to the same preview source.":`Generate with ${sourceLabel}.`;
}
function promptMeta(prompt){
  const values=[`${prompt.width||"—"} × ${prompt.height||"—"}`,`${prompt.steps||"—"} steps`,String(prompt.seed??"").trim()?`Seed ${prompt.seed}`:"Random seed"];
  if(String(prompt.guidance??"").trim())values.push(`Guidance ${prompt.guidance}`);
  if(String(prompt.frames??"").trim())values.push(`${prompt.frames} frames`);
  return values;
}
function movePlanItem(list,index,delta,openKey){
  const target=index+delta;if(target<0||target>=list.length)return;
  [list[index],list[target]]=[list[target],list[index]];state[openKey]=target;renderPlan();sync();
}
function renderPromptCards(){
  const host=$("#prompt-list"),prompts=state.settings.sample_prompts_data;
  host.innerHTML="";
  if(!prompts.length){
    host.innerHTML='<div class="plan-empty-state"><span>Aa</span><h3>Create a stable visual checkpoint</h3><p>Add two or three prompts you can compare throughout training. Keep the seed fixed when you want changes to reflect the LoRA rather than randomness.</p><div><button class="quiet" data-empty-library>Choose from library</button><button class="primary" data-empty-add>＋ Add first prompt</button></div></div>';
    host.querySelector("[data-empty-library]").addEventListener("click",()=>openPromptLibrary().catch(error=>toast(error.message,"error")));
    host.querySelector("[data-empty-add]").addEventListener("click",addPlanPrompt);
    return;
  }
  prompts.forEach((prompt,index)=>{
    const text=String(prompt.prompt||"").replace(/\s+/g," ").trim(),empty=!text,enabled=prompt.enabled!==false,hasPreview=state.promptPreview?.indices?.includes(index),previewing=hasPreview&&["starting","running"].includes(state.promptPreview.status),previewReady=hasPreview&&state.promptPreview.status==="completed",previewFailed=hasPreview&&["failed","stopped"].includes(state.promptPreview.status),cardIssues=promptCardIssues(prompt,index),previewIssues=promptCardIssues({...prompt,enabled:true},index,{includeDisabled:true}),previewKind=state.settings.training_mode==="MiniMax H3 (Experimental)"&&Number(prompt.frames??39)>1?"video":"preview";
    const status=previewing?"Previewing":previewReady?"Preview ready":previewFailed?"Preview failed":cardIssues.length?"Needs attention":enabled?"Included":"Off",linked=Boolean(prompt._library_id),useLora=promptPreviewUsesLora(prompt),modelLabel=(state.settings.training_mode||"Model").replace(" (Experimental)",""),previewSupported=["Krea 2","MiniMax H3 (Experimental)"].includes(state.settings.training_mode);
    const card=document.createElement("article");card.className=`plan-prompt-card${enabled?" is-included":" is-off"}${cardIssues.length?" needs-attention":""}${previewing?" is-previewing":""}${previewReady?" is-preview-ready":""}${previewFailed?" preview-failed":""}`;card.dataset.promptIndex=index;
    const previewPosition=state.promptPreview?.indices?.indexOf(index)??-1,previewPath=previewPosition>=0?state.promptPreview?.outputs?.[previewPosition]:"",previewUrl=previewPath?`/api/sample-file?path=${encodeURIComponent(previewPath)}`:"",previewIsVideo=/\.(mp4|webm|mov|m4v)$/i.test(previewPath||"");
    const thumbnail=previewUrl?(previewIsVideo?`<video src="${previewUrl}" muted loop playsinline preload="metadata" aria-label="Latest generated video preview"></video>`:`<img src="${previewUrl}" loading="lazy" alt="Latest generated preview">`):linked?`<img src="/api/prompt-library/thumbnail?id=${encodeURIComponent(prompt._library_id)}" loading="lazy" alt="Latest tested library preview">`:"";
    card.innerHTML=`<div class="prompt-card-visual">${thumbnail}<div class="prompt-visual-placeholder" aria-hidden="true"><span>Aa</span><small>${esc(`${prompt.width||"?"} × ${prompt.height||"?"}`)}</small></div><span class="prompt-status" aria-live="polite">${previewing?'<i class="status-spinner"></i>':""}${esc(status)}</span><span class="prompt-order">#${index+1}</span></div><div class="prompt-card-content"><div class="prompt-card-heading"><div><small>${linked?"LIBRARY PROMPT":"SAMPLE PROMPT"}</small><h3>${esc(prompt._library_name||`Sample prompt ${index+1}`)}</h3></div><details class="item-menu"><summary aria-label="More actions for sample prompt ${index+1}">•••</summary><div><button data-action="up" ${index===0?"disabled":""}>Move earlier</button><button data-action="down" ${index===prompts.length-1?"disabled":""}>Move later</button><button data-action="duplicate">Duplicate</button><button class="danger" data-action="remove">Remove</button></div></details></div><button class="prompt-card-copy" data-action="edit" aria-label="${esc(`Edit ${prompt._library_name||`sample prompt ${index+1}`}: ${(text||"empty prompt").slice(0,120)}`)}"><p>${esc(text||"Write the positive prompt Musubi should sample.")}</p>${prompt.neg?`<small>Negative: ${esc(String(prompt.neg).replace(/\s+/g," ").trim())}</small>`:""}</button><div class="prompt-meta">${promptMeta(prompt).map(value=>`<span>${esc(value)}</span>`).join("")}</div>${cardIssues.length?`<button class="plan-card-warning" data-action="edit"><span>!</span>${esc(cardIssues[0])}</button>`:""}<div class="prompt-card-footer"><label class="plan-switch"><input type="checkbox" data-action="enabled" ${enabled?"checked":""}><span>Include</span></label><div><button class="text-action" data-action="edit">Edit</button><button class="quiet" data-action="preview" title="${esc(previewIssues[0]||"Generate this prompt as a standalone preview.")}" ${!["Krea 2","MiniMax H3 (Experimental)"].includes(state.settings.training_mode)||previewIssues.length||previewing?"disabled":""}>${previewing?"Previewing…":previewReady?"Generate again":"Generate preview"}</button>${state.promptPreview?.jobId&&hasPreview?'<button class="text-action" data-action="view-run">View run</button>':""}</div></div></div>`;
    const footerActions=card.querySelector(".prompt-card-footer>div"),defaults=document.createElement("button"),source=document.createElement("label"),previewButton=card.querySelector('[data-action="preview"]');
    defaults.type="button";defaults.className="text-action";defaults.dataset.action="defaults";defaults.textContent=`Use ${modelLabel} defaults`;defaults.title=`Reset only this prompt's sampling values to the recommended ${modelLabel} settings. Prompt text is preserved.`;
    source.className="preview-card-toggle";source.title=useLora?"This card generates with the selected/current LoRA. Turn it off for the base model; use Edit to choose the LoRA file and strength.":"This card generates with the base model. Turn it on for the selected/current LoRA; use Edit to choose the file and strength.";source.innerHTML=`<input type="checkbox" data-action="preview-lora" ${useLora?"checked":""}><span>${useLora?"LoRA":"Base"}</span>`;
    if(previewSupported){previewButton.textContent=previewing?"Previewing…":previewReady?`Again: ${previewKind} with ${useLora?"LoRA":"base"}`:`Generate ${previewKind} with ${useLora?"LoRA":"base"}`;footerActions.prepend(defaults,source)}else footerActions.prepend(defaults);
    card.querySelector("img")?.addEventListener("error",event=>event.currentTarget.remove());
    const cardVideo=card.querySelector(".prompt-card-visual video");
    if(cardVideo){cardVideo.addEventListener("canplay",()=>cardVideo.play().catch(()=>{}),{once:true});cardVideo.addEventListener("error",()=>cardVideo.remove())}
    const visual=card.querySelector(".prompt-card-visual");
    if(previewUrl){
      visual.title="Double-click to open the full preview";
      visual.setAttribute("aria-label","Double-click to open the full generated preview");
      visual.addEventListener("dblclick",()=>openSamplePreview({media_kind:previewIsVideo?"video":"image",url:previewUrl,name:prompt._library_name||`Sample prompt ${index+1}`}));
    }
    card.querySelector('[data-action="enabled"]').addEventListener("change",event=>{prompt.enabled=event.target.checked;renderPlan();sync()});
    card.querySelector('[data-action="defaults"]').addEventListener("click",()=>applyPromptModelDefaults(index));
    card.querySelector('[data-action="preview-lora"]')?.addEventListener("change",event=>{prompt._preview_use_lora=event.target.checked;renderPlan();sync()});
    card.querySelectorAll('[data-action="edit"]').forEach(button=>button.addEventListener("click",()=>openPlanPromptEditor(index,button)));
    card.querySelector('[data-action="preview"]').addEventListener("click",()=>startPromptPreview([{...prompt,enabled:true}],{indices:[index],stayInPlan:true}));
    card.querySelector('[data-action="view-run"]')?.addEventListener("click",()=>go("run"));
    card.querySelector('[data-action="up"]').addEventListener("click",()=>movePlanItem(prompts,index,-1,"openPromptIndex"));
    card.querySelector('[data-action="down"]').addEventListener("click",()=>movePlanItem(prompts,index,1,"openPromptIndex"));
    card.querySelector('[data-action="duplicate"]').addEventListener("click",()=>{const copy=structuredClone(prompt);delete copy._library_id;delete copy._library_revision;copy._library_name=`${prompt._library_name||`Sample prompt ${index+1}`} copy`;prompts.splice(index+1,0,copy);renderPlan();sync()});
    card.querySelector('[data-action="remove"]').addEventListener("click",()=>{prompts.splice(index,1);state.openPromptIndex=-1;renderPlan();sync()});
    host.append(card);
  });
}
function planEditorSection(title,copy){
  const section=document.createElement("section");section.className="plan-editor-section";
  section.innerHTML=`<div class="plan-editor-section-head"><h3>${esc(title)}</h3>${copy?`<p>${esc(copy)}</p>`:""}</div><div class="guided-fields"></div>`;
  return section;
}
function resolutionPresets(){
  return state.settings.training_mode==="Wan 2.2"?[
    ["Square",512,512],["Landscape",832,480],["Portrait",480,832]
  ]:[
    ["Square",1024,1024],["Landscape",1216,832],["Portrait",832,1216]
  ];
}
function openPlanPromptEditor(index,trigger){
  state.openPromptIndex=index;state.planEditorReturnFocus=trigger;renderPlanPromptEditor();
  const dialog=$("#plan-prompt-dialog");if(!dialog.open)dialog.showModal();
}
function renderPlanPromptEditor(){
  const prompt=state.settings.sample_prompts_data?.[state.openPromptIndex];if(!prompt)return;
  const index=state.openPromptIndex,host=$("#plan-prompt-editor"),mode=state.settings.training_mode||"Wan 2.2";
  $("#plan-prompt-kicker").textContent=`SAMPLE PROMPT ${index+1} OF ${state.settings.sample_prompts_data.length}`;
  $("#plan-prompt-title").textContent=prompt._library_name||`Edit sample prompt ${index+1}`;
  $("#plan-prompt-editor-state").textContent=prompt.enabled===false?"Not included in scheduled samples":"Included in scheduled samples";
  host.innerHTML="";
  const words=planEditorSection("Prompt text","This is the content you will compare across checkpoints.");
  words.querySelector(".guided-fields").append(
    objectField("Positive prompt","prompt",prompt.prompt||"",value=>prompt.prompt=value,{type:"textarea",wide:true,help:"Describe the result Musubi should generate. Hover or focus this label for help."}),
    objectField("Negative prompt","neg",prompt.neg||"",value=>prompt.neg=value,{type:"textarea",wide:true,help:mode==="MiniMax H3 (Experimental)"?"MiniMax H3 does not use classifier-free guidance. Leave this blank.":"Optional concepts or defects to discourage in this comparison sample."})
  );
  const resolution=planEditorSection("Frame and composition","Use a preset or enter an exact size. The selected values are preserved in Musubi's prompt file.");
  const presets=document.createElement("div");presets.className="resolution-presets";
  const sizePresets=resolutionPresets(),matchedPreset=sizePresets.some(([,width,height])=>Number(prompt.width)===width&&Number(prompt.height)===height);
  sizePresets.forEach(([label,width,height])=>{const button=document.createElement("button");button.type="button";button.className=Number(prompt.width)===width&&Number(prompt.height)===height?"active":"";button.innerHTML=`<strong>${label}</strong><small>${width} × ${height}</small>`;button.addEventListener("click",()=>{prompt.width=width;prompt.height=height;renderPlanPromptEditor();sync()});presets.append(button)});
  const custom=document.createElement("button");custom.type="button";custom.className=`custom${matchedPreset?"":" active"}`;custom.innerHTML="<strong>Custom</strong><small>Exact dimensions</small>";custom.addEventListener("click",()=>resolution.querySelector('[data-object-key="width"]')?.focus());presets.append(custom);
  resolution.insertBefore(presets,resolution.querySelector(".guided-fields"));
  resolution.querySelector(".guided-fields").append(
    objectField("Width","width",prompt.width||"",value=>prompt.width=value,{type:"number",help:"Output width in pixels."}),
    objectField("Height","height",prompt.height||"",value=>prompt.height=value,{type:"number",help:"Output height in pixels."})
  );
  const sampling=planEditorSection("Sampling behavior","Keep the seed fixed for a direct checkpoint comparison, or leave it blank for a random seed.");
  sampling.querySelector(".guided-fields").append(
    objectField("Denoising steps","steps",prompt.steps||"",value=>prompt.steps=value,{type:"number",help:mode==="Krea 2"?"RAW commonly uses about 28 steps; Turbo commonly uses about 8.":mode==="MiniMax H3 (Experimental)"?"20 steps matches the current ComfyUI MiniMax H3 workflow default.":"Number of denoising steps for this sample."}),
    objectField("Guidance","guidance",prompt.guidance??"",value=>prompt.guidance=value,{type:"number",help:mode==="Krea 2"?"RAW commonly uses 5.5; Turbo usually uses 1.0.":mode==="MiniMax H3 (Experimental)"?"MiniMax H3 is guidance-distilled. Keep this at 1.0.":"Classifier-free guidance for this sample."}),
    objectField("Seed","seed",prompt.seed??"",value=>prompt.seed=value,{type:"number",help:"A fixed seed makes visual changes across checkpoints easier to attribute to training."})
  );
  const seedActions=document.createElement("div");seedActions.className="seed-actions";seedActions.innerHTML=`<span>Seed mode</span><button type="button" class="${String(prompt.seed??"").trim()?"active":""}" data-seed-mode="fixed">Fixed</button><button type="button" class="${String(prompt.seed??"").trim()?"":"active"}" data-seed-mode="random">Random</button>`;
  seedActions.querySelector('[data-seed-mode="fixed"]').addEventListener("click",()=>{if(!String(prompt.seed??"").trim())prompt.seed=42;renderPlanPromptEditor();sync()});
  seedActions.querySelector('[data-seed-mode="random"]').addEventListener("click",()=>{prompt.seed="";renderPlanPromptEditor();sync()});
  sampling.append(seedActions);
  host.append(words,resolution,sampling);
  if(["Krea 2","MiniMax H3 (Experimental)"].includes(mode)){
    const previewSource=planEditorSection("Standalone preview source","This affects Generate Preview for this card only. The selected LoRA file and strength are shared so every LoRA-enabled card compares the same checkpoint.");
    previewSource.querySelector(".guided-fields").append(
      objectField("Generate this card with","_preview_use_lora",promptPreviewUsesLora(prompt)?"lora":"base",value=>prompt._preview_use_lora=value==="lora",{type:"select",options:[{label:"Current / selected LoRA",value:"lora"},{label:"Base model only",value:"base"}],help:"Choose LoRA to test what training learned, or Base to create a control image without the LoRA."}),
      objectField("LoRA file (blank = automatic)","preview_lora_path",state.settings.preview_lora_path||"",value=>state.settings.preview_lora_path=value,{wide:true,browseKind:"file",help:"Leave blank to use the continuation LoRA or latest checkpoint matching this run. Browse to test a specific safetensors file."}),
      objectField("LoRA strength","preview_lora_multiplier",state.settings.preview_lora_multiplier||"1.0",value=>state.settings.preview_lora_multiplier=value,{type:"number",help:"1.0 applies the LoRA at its normal strength. This shared value is used by every prompt card set to LoRA."})
    );
    host.append(previewSource);
  }
  if(mode==="Krea 2"){
    const advanced=document.createElement("details");advanced.className="plan-editor-advanced";advanced.innerHTML='<summary>Standalone preview timestep controls</summary><p>Mu, Y1, and Y2 are honored by standalone Krea preview generation. Musubi training samples currently use the trainer’s RAW shift defaults.</p><div class="guided-fields"></div>';
    advanced.querySelector(".guided-fields").append(objectField("Mu","mu",prompt.mu??"",value=>prompt.mu=value,{type:"number",help:"Direct shift override for standalone preview."}),objectField("Y1","y1",prompt.y1??"",value=>prompt.y1=value,{type:"number",help:"Low-resolution shift endpoint for standalone preview."}),objectField("Y2","y2",prompt.y2??"",value=>prompt.y2=value,{type:"number",help:"High-resolution shift endpoint for standalone preview."}));
    host.append(advanced);
  }else{
    const advanced=document.createElement("details");advanced.className="plan-editor-advanced";advanced.open=mode==="Wan 2.2";advanced.innerHTML='<summary>Model-specific sample controls</summary><div class="guided-fields"></div>';
    const fields=advanced.querySelector(".guided-fields");
    if(mode==="Wan 2.2")fields.append(objectField("Frames","frames",prompt.frames??25,value=>prompt.frames=value,{type:"number",help:"Number of frames in a Wan sample."}));
    if(mode==="MiniMax H3 (Experimental)")fields.append(
      objectField("Frames","frames",prompt.frames??39,value=>prompt.frames=value,{type:"number",help:"Use 39 for the recommended short standalone preview matching ComfyUI's one-second setting. MiniMax H3 accepts 5, 22, 39, ... frames here. Scheduled in-training samples use the separate Scheduled MiniMax Preview option in Sampling Frequency."}),
      objectField("FPS","fps",prompt.fps??24,value=>prompt.fps=value,{type:"number",help:"Playback speed for the silent preview MP4. 24 FPS matches MiniMax H3's normal timing."})
    );
    fields.append(objectField("Flow shift","flow_shift",prompt.flow_shift??"",value=>prompt.flow_shift=value,{type:"number"}),objectField("CFG scale","cfg_scale",prompt.cfg_scale??"",value=>prompt.cfg_scale=value,{type:"number"}));
    if(mode==="Wan 2.2")fields.append(objectField("I2V source image","image_path",prompt.image_path||"",value=>prompt.image_path=value,{wide:true,browseKind:"file",help:"Optional starting image for Wan I2V samples."}));
    host.append(advanced);
  }
  const include=document.createElement("label");include.className="editor-include-switch";include.innerHTML=`<input type="checkbox" ${prompt.enabled!==false?"checked":""}><span><strong>Include in scheduled training samples</strong><small>Turning this off keeps the card and its settings without sending it to Musubi.</small></span>`;
  include.querySelector("input").addEventListener("change",event=>{prompt.enabled=event.target.checked;$("#plan-prompt-editor-state").textContent=prompt.enabled?"Included in scheduled samples":"Not included in scheduled samples";sync()});host.append(include);
  const refreshPreviewAction=()=>{const button=$("#plan-prompt-preview"),issues=promptCardIssues({...prompt,enabled:true},index,{includeDisabled:true}),useLora=promptPreviewUsesLora(prompt);button.disabled=!["Krea 2","MiniMax H3 (Experimental)"].includes(mode)||Boolean(issues.length);button.textContent=`Generate with ${useLora?"LoRA":"base"}`;button.title=issues[0]||`Generate this prompt with ${useLora?"the selected/current LoRA":"the base model only"}.`};
  host.oninput=refreshPreviewAction;host.onchange=refreshPreviewAction;refreshPreviewAction();
}
function stageHandoff(previous,current){
  if(!state.settings.use_staged_training)return "Saved draft · normal run stays active";
  if(!previous||previous.enabled===false||current.enabled===false)return "Disabled stages are skipped";
  return previous.type==="face_refinement"||current.type==="face_refinement"?"Complete LoRA handoff":"Exact training-state handoff";
}
function renderStageTimeline(){
  const host=$("#stage-list"),stages=state.settings.staged_training_config;host.innerHTML="";host.classList.toggle("is-disabled",!state.settings.use_staged_training);
  if(!stages.length){
    host.innerHTML='<div class="plan-empty-state stage-empty"><span>1→2</span><h3>One recipe is enough for most runs</h3><p>Add stages when you deliberately change dataset resolution or hand a finished LoRA into Face Refinement.</p><button class="primary" data-empty-stage>＋ Add first stage</button></div>';
    host.querySelector("[data-empty-stage]").addEventListener("click",addPlanStage);return;
  }
  stages.forEach((stage,index)=>{
    if(index>0){const connector=document.createElement("div");connector.className="stage-handoff";connector.innerHTML=`<i></i><span>${esc(stageHandoff(stages[index-1],stage))}</span>`;host.append(connector)}
    const limit=String(stage.steps||"").trim()?`${stage.steps} steps`:`${stage.epochs||"—"} epochs`,face=stage.type==="face_refinement",enabled=stage.enabled!==false,cardIssues=stageCardIssues(stage,index),draft=!state.settings.use_staged_training;
    const card=document.createElement("article");card.className=`plan-stage-card${enabled?" is-included":" is-off"}${cardIssues.length?" needs-attention":""}${draft?" is-draft":""}`;
    const overrides=[];if(stage.dop_mode&&stage.dop_mode!=="inherit")overrides.push(`DOP ${stage.dop_mode}`);if(stage.depth_helpers_mode&&stage.depth_helpers_mode!=="inherit")overrides.push(stage.depth_helpers_mode);
    const followsFace=index>0&&stages[index-1].enabled!==false&&stages[index-1].type==="face_refinement"&&!face;
    card.innerHTML=`<span class="stage-node">${index+1}</span><div class="stage-card-main"><div class="stage-card-heading"><div><small>${face?"FACE REFINEMENT":"STANDARD TRAINING"}</small><h3>${esc(stage.label||`Stage ${index+1}`)}</h3></div><span class="stage-state">${draft?(enabled?"Draft":"Draft · Off"):enabled?"Included":"Off"}</span></div><button class="stage-card-summary" data-stage-action="edit"><strong>${esc(limit)}</strong><span>${face?"Uses the Face Refinement recipe":esc(stage.dataset_config||"Choose a dataset TOML")}</span></button><div class="stage-card-meta"><span>Output · ${esc(`${state.settings.output_name||"run"}-${planStageArtifactLabel(stage,index)}`)}</span>${overrides.map(value=>`<span>${esc(value)}</span>`).join("")}</div>${cardIssues.length?`<button class="plan-card-warning" data-stage-action="edit"><span>!</span>${esc(cardIssues[0])}</button>`:""}${followsFace?'<div class="plan-card-note">This standard stage starts a new optimizer after face refinement; review whether that could soften identity gains.</div>':""}<div class="stage-card-footer"><label class="plan-switch"><input type="checkbox" data-stage-action="enabled" ${enabled?"checked":""}><span>Include</span></label><div><button class="text-action" data-stage-action="edit">Edit stage</button><details class="item-menu"><summary aria-label="More actions for stage ${index+1}">•••</summary><div><button data-stage-action="up" ${index===0?"disabled":""}>Move earlier</button><button data-stage-action="down" ${index===stages.length-1?"disabled":""}>Move later</button><button data-stage-action="duplicate">Duplicate</button><button class="danger" data-stage-action="remove">Remove</button></div></details></div></div></div>`;
    card.querySelectorAll('[data-stage-action="edit"]').forEach(button=>button.addEventListener("click",()=>openStageEditor(index,button)));
    card.querySelector('[data-stage-action="enabled"]').addEventListener("change",event=>{stage.enabled=event.target.checked;renderPlan();sync()});
    card.querySelector('[data-stage-action="up"]').addEventListener("click",()=>movePlanItem(stages,index,-1,"openStageIndex"));card.querySelector('[data-stage-action="down"]').addEventListener("click",()=>movePlanItem(stages,index,1,"openStageIndex"));
    card.querySelector('[data-stage-action="duplicate"]').addEventListener("click",()=>{const copy=structuredClone(stage);copy.label=uniqueStageLabel(`${stage.label||`Stage ${index+1}`} copy`);stages.splice(index+1,0,copy);state.openStageIndex=index+1;renderPlan();sync()});
    card.querySelector('[data-stage-action="remove"]').addEventListener("click",()=>{stages.splice(index,1);state.openStageIndex=-1;renderPlan();sync()});host.append(card);
  });
  const add=document.createElement("button");add.type="button";add.className="stage-add-node";add.innerHTML="<span>＋</span><strong>Add another stage</strong>";add.addEventListener("click",addPlanStage);host.append(add);
}
function openStageEditor(index,trigger){
  state.openStageIndex=index;state.stageEditorReturnFocus=trigger;renderStageEditor();
  const dialog=$("#stage-editor-dialog");if(!dialog.open)dialog.showModal();
}
function renderStageEditor(){
  const stage=state.settings.staged_training_config?.[state.openStageIndex];if(!stage)return;
  const index=state.openStageIndex,host=$("#stage-editor-body"),mode=state.settings.training_mode||"Wan 2.2",face=stage.type==="face_refinement";
  $("#stage-editor-kicker").textContent=`STAGE ${index+1} OF ${state.settings.staged_training_config.length}`;$("#stage-editor-title").textContent=stage.label||`Stage ${index+1}`;$("#stage-editor-state").textContent=stage.enabled===false?"Not included in the staged run":"Included in the staged run";host.innerHTML="";
  const identity=planEditorSection("Stage identity","The label becomes part of the output and state-folder name.");
  identity.querySelector(".guided-fields").append(
    objectField("Label","label",stage.label||`Stage ${index+1}`,value=>{stage.label=value;$("#stage-editor-title").textContent=value||`Stage ${index+1}`},{help:"Use a unique Windows-safe label. Numeric labels automatically gain “px” in artifact names."}),
    objectField("Stage type","type",stage.type||"standard",value=>{stage.type=value;if(value==="face_refinement"){stage.steps=stage.steps||faceConfig().steps||30;stage.epochs=""}renderStageEditor();sync()},{type:"select",options:[{label:"Standard training",value:"standard"},{label:"Face refinement",value:"face_refinement"}],help:"Standard stages pass exact Accelerate state. Face Refinement consumes and emits a complete LoRA."})
  );host.append(identity);
  const schedule=planEditorSection(face?"Refinement limit":"Dataset and limit",face?"Face Refinement always uses a step limit.":"Choose either epochs or steps. Entering one clears the other.");
  const scheduleFields=schedule.querySelector(".guided-fields");
  if(face){
    scheduleFields.append(objectField("Refinement steps","steps",stage.steps||faceConfig().steps||30,value=>stage.steps=value,{type:"number",help:"Uses the Face Refinement recipe and analyzed references."}));
    const link=document.createElement("button");link.type="button";link.className="workspace-link compact";link.innerHTML="<span>◎</span><span><strong>Open Face Refinement</strong><small>Review references, reward, and evaluation settings</small></span><b>→</b>";link.addEventListener("click",()=>{$("#stage-editor-dialog").close();go("face")});schedule.append(link);
  }else{
    scheduleFields.append(objectField("Dataset TOML","dataset_config",stage.dataset_config||state.settings.dataset_config||"",value=>stage.dataset_config=value,{wide:true,browseKind:"file",help:"This stage's own Musubi dataset configuration."}));
    const limitMode=String(stage.steps||"").trim()?"steps":"epochs",chooser=document.createElement("div");chooser.className="stage-limit-chooser";chooser.innerHTML=`<span>Train by</span><button type="button" data-limit="epochs" class="${limitMode==="epochs"?"active":""}">Epochs</button><button type="button" data-limit="steps" class="${limitMode==="steps"?"active":""}">Steps</button>`;
    chooser.querySelectorAll("[data-limit]").forEach(button=>button.addEventListener("click",()=>{if(button.dataset.limit==="steps"){stage.steps=stage.steps||1;stage.epochs=""}else{stage.epochs=stage.epochs||1;stage.steps=""}renderStageEditor();sync()}));schedule.insertBefore(chooser,scheduleFields);
    scheduleFields.append(limitMode==="steps"?objectField("Maximum steps","steps",stage.steps||1,value=>{stage.steps=value;stage.epochs=""},{type:"number"}):objectField("Epochs","epochs",stage.epochs||1,value=>{stage.epochs=value;stage.steps=""},{type:"number"}));
  }
  host.append(schedule);
  if(!face&&["Krea 2","Flux.2 Klein","MiniMax H3 (Experimental)"].includes(mode)){
    const advanced=document.createElement("details");advanced.className="plan-editor-advanced";advanced.innerHTML='<summary>Stage regularization overrides</summary><p>Inherit follows the main recipe. An explicit setting affects only this stage.</p><div class="guided-fields"></div>';
    const fields=advanced.querySelector(".guided-fields");fields.append(objectField("DOP behavior","dop_mode",stage.dop_mode||"inherit",value=>stage.dop_mode=value,{type:"select",options:[{label:"Inherit main recipe",value:"inherit"},{label:"Enable for this stage",value:"enable"},{label:"Disable for this stage",value:"disable"}]}),objectField("DOP strength","dop_loss_weight",stage.dop_loss_weight||"",value=>stage.dop_loss_weight=value,{type:"number"}),objectField("DOP trigger word","dop_trigger_word",stage.dop_trigger_word||"",value=>stage.dop_trigger_word=value),objectField("DOP class word","dop_class_word",stage.dop_class_word||"",value=>stage.dop_class_word=value));
    if(["Krea 2","MiniMax H3 (Experimental)"].includes(mode))fields.append(objectField("Depth helper memory","depth_helpers_mode",stage.depth_helpers_mode||"inherit",value=>stage.depth_helpers_mode=value,{type:"select",options:[{label:"Inherit main recipe",value:"inherit"},{label:"Keep on GPU",value:"keep on GPU"},{label:"Offload to CPU",value:"offload to CPU"}]}));host.append(advanced);
  }
  const include=document.createElement("label");include.className="editor-include-switch";include.innerHTML=`<input type="checkbox" ${stage.enabled!==false?"checked":""}><span><strong>Include this stage</strong><small>Disabled stages stay in the plan but are skipped during training.</small></span>`;
  include.querySelector("input").addEventListener("change",event=>{stage.enabled=event.target.checked;$("#stage-editor-state").textContent=stage.enabled?"Included in the staged run":"Not included in the staged run";sync()});host.append(include);
}
function addPlanPrompt(){
  const prompts=state.settings.sample_prompts_data||(state.settings.sample_prompts_data=[]);prompts.push(defaultPromptForMode());state.openPromptIndex=prompts.length-1;renderPlan();sync();
  requestAnimationFrame(()=>openPlanPromptEditor(state.openPromptIndex,$(`.plan-prompt-card[data-prompt-index="${state.openPromptIndex}"] [data-action="edit"]`)));
}
function addPlanStage(){
  const stages=state.settings.staged_training_config||(state.settings.staged_training_config=[]),label=uniqueStageLabel(`Stage ${stages.length+1}`);
  stages.push({enabled:true,type:"standard",label,dataset_config:state.settings.dataset_config||"",epochs:"1",steps:"",dop_mode:"inherit",depth_helpers_mode:"inherit"});state.openStageIndex=stages.length-1;renderPlan();sync();
  requestAnimationFrame(()=>openStageEditor(state.openStageIndex,$$(".plan-stage-card").at(-1)?.querySelector('[data-stage-action="edit"]')));
}
function renderPlan(){
  state.settings.sample_prompts_data=Array.isArray(state.settings.sample_prompts_data)?state.settings.sample_prompts_data:[];
  state.settings.staged_training_config=Array.isArray(state.settings.staged_training_config)?state.settings.staged_training_config:[];
  $("#use-stages").checked=Boolean(state.settings.use_staged_training);
  $("#normal-cache-policy").hidden=Boolean(state.settings.use_staged_training);
  $("#staged-cache-policy").hidden=!state.settings.use_staged_training;
  appendFields($("#run-cache-policies"),["recache_latents","recache_text"]);
  appendFields($("#stage-policies"),["staged_recache_latents","staged_recache_text"]);
  appendFields($("#sampling-frequency-fields"),["sample_every_n_epochs","sample_every_n_steps","sample_at_first","minimax_h3_training_preview_mode"]);
  appendFields($("#checkpoint-frequency-fields"),["save_every_n_epochs","save_every_n_steps","rename_final_artifacts_to_epoch"]);
  appendFields($("#notes-fields"),["training_comment","auto_training_settings_summary"]);
  ensurePreviewSettings();renderSamplingEstimate();renderPromptCards();renderStageTimeline();renderPlanOverview();renderTrainingSummary();
}
async function saveTrainingPlan(){
  const issues=promptPlanIssues();renderPlanOverview();
  if(issues.length){toast(`Fix the plan before saving: ${issues[0]}`,"error");return false}
  const saved=await saveSettings();
  if(saved)renderPlanOverview();
  return saved;
}
function clientTrainingSettingsSummary(){
  const settings=state.settings,parts=[settings.training_mode||"Training"],output=String(settings.output_name||"").trim();
  if(output)parts.push(`run=${output}`);
  let network=String(settings.network_type||"LoRA"),rank=String(settings.network_dim_low||"").trim(),alpha=String(settings.network_alpha_low||"").trim();
  if(rank)network+=` rank ${rank}`;if(alpha)network+=` α${alpha}`;parts.push(network);
  const stages=(settings.staged_training_config||[]).filter(stage=>stage.enabled!==false);
  if(settings.use_staged_training&&stages.length&&!settings.stage_type){
    parts.push(`staged ${stages.map((stage,index)=>`${planStageArtifactLabel(stage,index)} ${String(stage.steps||"").trim()?`${stage.steps} steps`:`${stage.epochs||"?"} epochs`}${stage.type==="face_refinement"?" face":""}`).join(" → ")}`);
  }else{
    const steps=String(settings.max_train_steps||"").trim(),epochs=String(settings.max_train_epochs||"").trim(),dataset=String(settings.dataset_config||"").split(/[\\/]/).pop()?.replace(/\.[^.]+$/,"");
    if(steps)parts.push(`${steps} steps`);else if(epochs)parts.push(`${epochs} epochs`);
    if(dataset)parts.push(`data=${dataset}`);
  }
  if(String(settings.learning_rate||"").trim())parts.push(`lr=${settings.learning_rate}`);
  if(String(settings.optimizer_type||"").trim())parts.push(`opt=${settings.optimizer_type}`);
  if(settings.dop_enabled){
    let dop=`DOP ${Number(settings.dop_loss_weight)>0?Number(settings.dop_loss_weight):"?"}`;
    if(String(settings.dop_class_word||"").trim())dop+=` (${settings.dop_class_word})`;parts.push(dop);
  }
  if(["Krea 2","MiniMax H3 (Experimental)"].includes(settings.training_mode)){
    if(Number(settings.krea2_depth_anchor_weight)>0)parts.push(`depth ${Number(settings.krea2_depth_anchor_weight)}@${settings.krea2_depth_anchor_input_size||518}${settings.krea2_keep_depth_helpers_on_gpu?" GPU":" offload"}`);
    if(Number(settings.krea2_weight_noise_sigma)>0)parts.push(`weight-noise ${Number(settings.krea2_weight_noise_sigma)} ${settings.krea2_weight_noise_mode||"relative"}`);
    if(settings.training_mode==="Krea 2"&&String(settings.krea2_projector_diff||"").trim())parts.push(`projector=${String(settings.krea2_projector_diff).split(/[\\/]/).pop()}@${settings.krea2_projector_diff_strength||1}`);
  }
  const blocks=String(settings.blocks_to_swap||"").trim();if(blocks&&blocks!=="0")parts.push(`swap=${blocks}`);
  const saveEpochs=String(settings.save_every_n_epochs||"").trim(),saveSteps=String(settings.save_every_n_steps||"").trim();
  if(saveEpochs&&saveEpochs!=="0")parts.push(`save every ${saveEpochs} epoch${Number(saveEpochs)===1?"":"s"}`);
  if(saveSteps&&saveSteps!=="0")parts.push(`save every ${saveSteps} steps`);
  return `Settings: ${parts.join("; ")}`;
}
function renderSamplingEstimate(){
  const host=$("#sampling-epoch-estimate");if(!host)return;
  const estimate=state.samplingEstimate;
  if(!estimate)host.innerHTML='<span><strong>Estimated steps per epoch</strong><small>Load or audit the dataset to calculate this.</small></span><button class="quiet" id="estimate-epoch-steps" type="button">Estimate from dataset</button>';
  else{
    const steps=Number(estimate.steps_per_epoch||0),samples=Number(estimate.effective_samples||0),batches=Number(estimate.batches_per_epoch||0),fraction=String(state.settings.sample_every_n_epochs||"").trim();
    const implied=fraction&&fraction!=="0"&&!Number.isInteger(Number(fraction))&&steps?Math.max(1,Math.round(steps*Number(fraction))):0;
    host.innerHTML=`<span><strong>${steps?`Estimated ${steps.toLocaleString()} optimizer steps per epoch`:"No usable samples found"}</strong><small>${samples.toLocaleString()} effective samples · ${batches.toLocaleString()} batches${implied?` · ${implied.toLocaleString()} steps for every ${fraction} epoch`:""}</small></span><button class="quiet" id="estimate-epoch-steps" type="button">Recalculate</button>`;
  }
  $("#estimate-epoch-steps")?.addEventListener("click",estimateSamplingSteps);
}
async function estimateSamplingSteps(){
  const path=String($("#dataset-path")?.value||state.settings.dataset_config||"").trim();
  if(!path){toast("Choose a dataset TOML before estimating epoch steps.","error");return}
  try{
    await flushDatasetDraft();
    const payload=await api("/api/dataset/estimate-steps",{method:"POST",body:JSON.stringify({path,text:$("#dataset-source")?.value||"",gradient_accumulation_steps:state.settings.gradient_accumulation_steps||1})});
    state.samplingEstimate=payload;renderSamplingEstimate();
  }catch(error){state.samplingEstimate=null;renderSamplingEstimate();toast(error.message,"error")}
}
function renderTrainingSummary(){
  const summary=clientTrainingSettingsSummary();
  $("#training-summary").textContent=state.settings.auto_training_settings_summary===false?`Automatic summary is off.\n\nPreview: ${summary}`:summary;
}
let libraryEntries=[];
async function openPromptLibrary(){
  const payload=await api("/api/prompt-library");libraryEntries=payload.prompts||[];renderLibrary();if(!$("#prompt-library-dialog").open)$("#prompt-library-dialog").showModal();
}
async function startPromptPreview(prompts,{indices=[],stayInPlan=true}={}){
  const sources=new Set(prompts.map(prompt=>promptPreviewUsesLora(prompt)));
  if(sources.size>1)return toast("These cards mix Base and LoRA preview sources. Generate them individually, or set them to the same source first.","error");
  const previewSettings={...state.settings,preview_use_lora:sources.values().next().value??state.settings.preview_use_lora};
  state.promptPreview={status:"starting",indices:[...indices],jobId:null,message:"Preparing preview"};renderPlan();
  try{
    const payload=await api("/api/prompts/preview",{method:"POST",body:JSON.stringify({settings:previewSettings,prompts})});
    state.promptPreview={status:"running",indices:[...indices],jobId:payload.job?.id||null,message:"Generating preview",savePath:payload.save_path};
    lastLogId=0;renderActive(payload.job);renderPlan();
    if(!stayInPlan)go("run");
    const source=payload.network_weights?`LoRA ${String(payload.network_weights).split(/[\\/]/).pop()} at ${payload.lora_multiplier||"1.0"}×`:"the base model (LoRA off)";
    toast(`Preview started with ${source}. You can keep editing this plan.`);
  }catch(e){
    state.promptPreview={status:"failed",indices:[...indices],jobId:null,message:e.message};renderPlan();toast(e.message,"error");
  }
}
function promptIdentityClient(prompt){
  const normalized={};
  Object.keys(prompt||{}).filter(key=>key!=="enabled"&&!key.startsWith("_library_")&&!key.startsWith("_preview_")).sort().forEach(key=>{const value=prompt[key];normalized[key]=typeof value==="string"?value.trim():value});
  return JSON.stringify(normalized);
}
function renderLibrary(){
  const query=($("#library-search").value||"").toLowerCase(),host=$("#library-list"),entries=libraryEntries.filter(entry=>JSON.stringify([entry.name,entry.tags,entry.collection,entry.prompt_data?.prompt]).toLowerCase().includes(query));
  host.innerHTML=entries.length?"":'<div class="empty">No matching library prompts.</div>';
  entries.sort((a,b)=>Number(Boolean(b.favorite))-Number(Boolean(a.favorite))||String(a.name).localeCompare(String(b.name))).forEach(entry=>{const card=document.createElement("article");card.className="structured-item library-card";card.innerHTML=`${entry.thumbnails?.length?`<img class="library-thumbnail" src="/api/prompt-library/thumbnail?id=${encodeURIComponent(entry.id)}" loading="lazy" alt="Latest tested result">`:""}<div class="library-copy"><div class="structured-item-head"><div><strong>${esc(entry.favorite?"★ ":"")}${esc(entry.name||"Untitled prompt")}</strong><small>${esc([entry.collection,...(entry.tags||[])].filter(Boolean).join(" · "))}</small></div><div class="structured-item-actions"><button data-library-action="favorite" title="Favorite">${entry.favorite?"★":"☆"}</button><button data-library-action="edit">Edit</button><button class="primary" data-library-action="add">Add to plan</button></div></div><p class="library-prompt-copy">${esc(entry.prompt_data?.prompt||"")}</p></div>`;
    card.querySelector(".library-thumbnail")?.addEventListener("error",event=>event.currentTarget.remove());
    card.querySelector('[data-library-action="add"]').addEventListener("click",()=>{const prompt=structuredClone(entry.prompt_data||{}),identity=promptIdentityClient(prompt),existing=(state.settings.sample_prompts_data||[]).findIndex(item=>promptIdentityClient(item)===identity);if(existing>=0){state.openPromptIndex=existing;toast("That prompt is already in this plan.");return}prompt._library_id=entry.id;prompt._library_revision=entry.revision||1;prompt._library_name=entry.name||"";prompt.enabled=true;state.settings.sample_prompts_data.push(prompt);state.openPromptIndex=state.settings.sample_prompts_data.length-1;renderPlan();sync();toast("Prompt added to this training plan.")});
    card.querySelector('[data-library-action="favorite"]').addEventListener("click",async()=>{await api("/api/prompt-library/favorite",{method:"POST",body:JSON.stringify({id:entry.id})});await openPromptLibrary()});
    card.querySelector('[data-library-action="edit"]').addEventListener("click",()=>openLibraryEditor(entry));host.append(card)})
}
function openLibraryEditor(entry){$("#library-edit-id").value=entry.id;$("#library-edit-name").value=entry.name||"";$("#library-edit-prompt").value=entry.prompt_data?.prompt||"";$("#library-edit-collection").value=entry.collection||"";$("#library-edit-tags").value=(entry.tags||[]).join(", ");$("#prompt-editor-dialog").showModal()}
function faceConfig(){state.settings.face_refinement_config=state.settings.face_refinement_config&&typeof state.settings.face_refinement_config==="object"?state.settings.face_refinement_config:{};return state.settings.face_refinement_config}
function updateFacePromptModeLabels(config){
  const poseActive=Boolean(config.pose_aware&&config.pose_plan?.enabled);
  $("#pose-plan-title").textContent="Per-pose targets and stopping";
  $("#pose-plan-help").innerHTML=poseActive?"Pose-aware training is on. Set each angle's share, identity target, and stopping rules in the table; edit its training prompts in the matching tab below.":"These rows are prepared but are not used until Pose-aware training is enabled. For a normal all-angle refinement, use the general prompts below.";
  $("#pose-prompts-help").innerHTML=poseActive?"These are the prompts used for this run. Select an angle tab, then edit one prompt per line; keep the matching <code>[angle]</code> tag at the start.":"Enable Pose-aware training to use separate prompts for each viewing angle.";
  const fallbackPrompts=$("#face-fallback-prompts");fallbackPrompts.open=!poseActive;fallbackPrompts.classList.toggle("is-inactive",poseActive);
  $("#face-prompts-title").textContent=poseActive?"General refinement prompts (fallback)":"General refinement prompts";
  $("#face-prompts-help").textContent=poseActive?"Kept for later; not used while the pose plan is active.":"Used for normal all-angle refinement when the pose plan is off.";
}
function renderPosePlanTable(host,config,buckets){
  const labels={frontal:"Frontal",three_quarter_left:"Three-quarter left",three_quarter_right:"Three-quarter right",profile_left:"Profile left",profile_right:"Profile right",looking_up:"Looking up",looking_down:"Looking down"};
  const order=["frontal","three_quarter_left","three_quarter_right","profile_left","profile_right","looking_up","looking_down"];
  const counts=config.preflight_report?.pose_bucket_counts||{};
  host.className="pose-plan-table-wrap";
  host.innerHTML=`<div class="pose-plan-table" role="table" aria-label="Pose training plan"><div class="pose-plan-row pose-plan-header" role="row"><span>Use</span><span>Viewing angle</span><span>Refs</span><span>Share %</span><span>Target</span><span>Target patience</span><span>Plateau patience</span><span>Min evaluations</span></div></div>`;
  const table=host.querySelector(".pose-plan-table");
  order.forEach(name=>{
    const bucket=buckets[name]||{},count=Number(counts[name]||0),label=labels[name]||name.replaceAll("_"," ");
    const row=document.createElement("div");row.className=`pose-plan-row${bucket.enabled===false?" is-disabled":""}`;row.dataset.pose=name;row.setAttribute("role","row");
    row.innerHTML=`<label class="pose-plan-use"><input type="checkbox" data-pose-enabled ${bucket.enabled!==false?"checked":""}><span class="sr-only">Use ${esc(label)}</span></label><div class="pose-plan-name"><strong>${esc(label)}</strong><small>${esc(name)}</small></div><span class="pose-plan-refs ${count<Number(config.pose_min_references??2)?"is-low":""}" title="${esc(`${count} reference face${count===1?"":"s"} assigned to this angle`)}">${count}</span>${[["share",bucket.share??0,"0","100","0.1"],["target",bucket.target??.55,"0","1","0.01"],["patience",bucket.patience??2,"0","","1"],["plateau_patience",bucket.plateau_patience??4,"0","","1"],["min_evaluations",bucket.min_evaluations??2,"1","","1"]].map(([key,value,min,max,step])=>`<input class="pose-plan-input" type="number" min="${min}"${max?` max="${max}"`:""} step="${step}" data-pose-field="${key}" value="${esc(String(Math.round(Number(value)*100)/100))}" aria-label="${esc(`${label} ${key.replaceAll("_"," ")}`)}">`).join("")}`;
    row.querySelector("[data-pose-enabled]").addEventListener("change",event=>{bucket.enabled=event.target.checked;row.classList.toggle("is-disabled",!event.target.checked);sync()});
    row.querySelectorAll("[data-pose-field]").forEach(input=>input.addEventListener("change",event=>{const key=event.target.dataset.poseField;const value=Number(event.target.value);if(Number.isFinite(value))bucket[key]=value;sync()}));
    table.append(row);
  });
}
function renderPosePromptEditor(host,config,buckets){
  const labels={frontal:"Frontal",three_quarter_left:"Three-quarter left",three_quarter_right:"Three-quarter right",profile_left:"Profile left",profile_right:"Profile right",looking_up:"Looking up",looking_down:"Looking down"};
  const order=["frontal","three_quarter_left","three_quarter_right","profile_left","profile_right","looking_up","looking_down"];
  const active=order.includes(state.facePosePromptTab)?state.facePosePromptTab:(order.find(name=>buckets[name]?.enabled!==false)||order[0]);
  state.facePosePromptTab=active;
  const bucket=buckets[active]||{},label=labels[active]||active.replaceAll("_"," "),prompts=Array.isArray(bucket.prompts)?bucket.prompts:[],count=Number(config.preflight_report?.pose_bucket_counts?.[active]||0);
  host.innerHTML=`<div class="pose-prompt-tabs" role="tablist" aria-label="Pose prompt tabs">${order.map(name=>{const item=buckets[name]||{},itemLabel=labels[name]||name.replaceAll("_"," "),itemPrompts=Array.isArray(item.prompts)?item.prompts:[];return `<button type="button" role="tab" aria-selected="${name===active}" class="${name===active?"active":""}" data-pose-prompt-tab="${name}"><span>${esc(itemLabel)}</span><small>${itemPrompts.length}</small></button>`}).join("")}</div><div class="pose-prompt-panel" role="tabpanel"><div class="pose-prompt-panel-head"><div><strong>${esc(label)} prompts</strong><small>${count} matching reference${count===1?"":"s"} · one prompt per line</small></div><button type="button" class="quiet" data-pose-ideas>Add suggested prompts</button></div><textarea data-pose-prompts aria-label="${esc(`${label} pose prompts`)}" placeholder="[${esc(active)}] ${esc(label.toLowerCase())} portrait of {trigger}, natural daylight">${esc(prompts.join("\n"))}</textarea><p class="pose-prompt-note">Keep <code>[${esc(active)}]</code> at the start of each line. This tag sends the prompt to the matching viewing-angle references.</p></div>`;
  host.querySelectorAll("[data-pose-prompt-tab]").forEach(button=>button.addEventListener("click",()=>{state.facePosePromptTab=button.dataset.posePromptTab;renderPosePromptEditor(host,config,buckets)}));
  host.querySelector("[data-pose-prompts]").addEventListener("change",event=>{bucket.prompts=event.target.value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean);renderPosePromptEditor(host,config,buckets);sync()});
  host.querySelector("[data-pose-ideas]").addEventListener("click",()=>{const phrases={frontal:"front-facing portrait",three_quarter_left:"three-quarter portrait, turned slightly left",three_quarter_right:"three-quarter portrait, turned slightly right",profile_left:"clear left side-profile portrait",profile_right:"clear right side-profile portrait",looking_up:"portrait looking slightly upward",looking_down:"portrait looking slightly downward"},suffixes={natural:"natural daylight, realistic skin texture",studio:"neutral studio background, soft balanced lighting",cinematic:"cinematic lighting, detailed photograph",expression:"natural expression, candid photograph"},trigger=config.trigger_word||"{trigger}",existing=new Set(bucket.prompts||[]);(config.pose_plan?.variations||["natural","studio","cinematic","expression"]).forEach(style=>{if(suffixes[style])existing.add(`[${active}] ${phrases[active]} of ${trigger}, ${suffixes[style]}`)});bucket.prompts=[...existing];renderPosePromptEditor(host,config,buckets);sync()});
}
function renderFaceWorkspace(){
  const config=faceConfig(),set=(key,value)=>{config[key]=value},isH3=state.settings.training_mode==="MiniMax H3 (Experimental)",supportsFace=["Krea 2","MiniMax H3 (Experimental)"].includes(state.settings.training_mode);
  config.pose_plan ||= {enabled:false,preset:"balanced_identity",overall_anchor_weight:.8,variations:["natural","studio","cinematic","expression"],buckets:{}};
  ["frontal","three_quarter_left","three_quarter_right","profile_left","profile_right","looking_up","looking_down"].forEach(name=>{config.pose_plan.buckets[name]||={enabled:true,share:14.286,target:.55,patience:2,plateau_patience:4,min_evaluations:2,prompts:[]}});
  updateFacePromptModeLabels(config);
  $("#face-mode-warning").style.display=supportsFace?"none":"";
  const evaluationTab=$('[data-face-step="evaluation"]');
  evaluationTab.disabled=isH3;
  evaluationTab.title=isH3?"Fixed Turbo evaluation is Krea 2-only; MiniMax H3 refinement itself is available.":"";
  const setup=$("#face-setup-fields");setup.innerHTML="";setup.append(objectField("Starting LoRA source","input_mode",config.input_mode||"previous_stage",v=>set("input_mode",v),{type:"select",options:[{label:"Use LoRA from previous stage",value:"previous_stage"},{label:"Refine an existing LoRA",value:"existing_lora"}]}),objectField("Input LoRA","input_lora",config.input_lora||state.settings.network_weights||"",v=>set("input_lora",v),{wide:true,browseKind:"file",help:"Required when refinement is the first stage; later face stages can consume the previous stage automatically."}),objectField("Trigger word","trigger_word",config.trigger_word||"",v=>set("trigger_word",v),{help:"The unique subject token used by reference and evaluation prompts."}),objectField("Reference directory","reference_dir",config.reference_dir||"",v=>set("reference_dir",v),{wide:true,browseKind:"directory",help:"Folder containing clear reference photographs of the person whose identity should be preserved."}),objectField("AntelopeV2 model folder","face_model_dir",config.face_model_dir||"",v=>set("face_model_dir",v),{wide:true,browseKind:"directory",help:"Click Browse and choose the AntelopeV2 folder itself. Existing InsightFace folders with glintr100.onnx and scrfd_10g_bnkps.onnx work, as do folders downloaded by this GUI."}),objectField("I acknowledge the AntelopeV2 model terms","license_acknowledged",config.license_acknowledged||false,v=>set("license_acknowledged",v),{type:"boolean",help:"Required only before this GUI downloads third-party model files. It is not needed merely to select models you already have."}));
  const modelFolderHelp=document.createElement("div");modelFolderHelp.className="plan-card-note";modelFolderHelp.innerHTML="<strong>Already have InsightFace or ReActor models?</strong> Use Browse and select the folder containing <code>glintr100.onnx</code> and <code>scrfd_10g_bnkps.onnx</code>. Select the folder, not an individual ONNX file. Otherwise, leave the default folder selected, acknowledge the model terms, and use the download button below.";setup.append(modelFolderHelp);
  const recipe=$("#face-recipe-fields");recipe.innerHTML="";recipe.className="face-recipe-groups";
  const addRecipeGroup=(title,copy,controls,open=false)=>{
    const group=document.createElement("details");group.className="recipe-group";group.open=open;
    group.innerHTML=`<summary><span><strong>${esc(title)}</strong><small>${esc(copy)}</small></span><b>⌄</b></summary><div class="guided-fields"></div>`;
    group.querySelector(".guided-fields").append(...controls);recipe.append(group);
  };
  addRecipeGroup("Core schedule","How long and how strongly to refine the adapter.",[
    objectField("Training steps","steps",config.steps??30,v=>set("steps",v),{type:"number"}),
    objectField("Resolution","resolution",config.resolution??512,v=>set("resolution",v),{type:"number"}),
    objectField("Learning rate","learning_rate",config.learning_rate??1e-4,v=>set("learning_rate",v),{type:"number"}),
    objectField("Denoising steps","denoise_steps",config.denoise_steps??12,v=>set("denoise_steps",v),{type:"number"}),
    objectField("DRaFT truncation K","draft_k",config.draft_k??1,v=>set("draft_k",v),{type:"number"}),
    objectField("CFG scale","cfg_scale",isH3?1:(config.cfg_scale??5.5),v=>set("cfg_scale",v),{type:"number",help:isH3?"MiniMax H3 is guidance-distilled, so this is fixed at 1.0.":"Krea 2 face-refinement classifier-free guidance."})
  ],true);
  addRecipeGroup("Quality and stopping","Identity targets, anti-copy pressure, and automatic stopping.",[
    objectField("Target similarity","target_similarity",config.target_similarity??.45,v=>set("target_similarity",v),{type:"number"}),
    objectField("Stop similarity","stop_similarity",config.stop_similarity??.55,v=>set("stop_similarity",v),{type:"number"}),
    objectField("Early-stop patience","early_stop_patience",config.early_stop_patience??5,v=>set("early_stop_patience",v),{type:"number"}),
    objectField("Minimum detection rate","min_detection_rate",config.min_detection_rate??.25,v=>set("min_detection_rate",v),{type:"number"}),
    objectField("Anti-copy weight","anti_copy_weight",config.anti_copy_weight??.02,v=>set("anti_copy_weight",v),{type:"number"})
  ]);
  addRecipeGroup("Pose guidance","How pose buckets affect the identity reward.",[
    objectField("Pose aware","pose_aware",config.pose_aware||false,v=>{set("pose_aware",v);updateFacePromptModeLabels(config)},{type:"boolean",help:"When on, training uses only the prompts in the enabled viewing-angle rows. Turn it off to use the general fallback prompts instead."}),
    objectField("Pose reward weight","pose_reward_weight",config.pose_reward_weight??.2,v=>set("pose_reward_weight",v),{type:"number"}),
    objectField("Minimum pose references","pose_min_references",config.pose_min_references??2,v=>set("pose_min_references",v),{type:"number"}),
    objectField("Overall identity anchor","pose_anchor",config.pose_plan?.overall_anchor_weight??.8,v=>{config.pose_plan.overall_anchor_weight=v},{type:"number"}),
    objectField("Prompt idea styles","pose_variations",(config.pose_plan?.variations||["natural","studio","cinematic","expression"]).join(", "),v=>{config.pose_plan.variations=String(v).split(",").map(x=>x.trim().toLowerCase()).filter(Boolean)},{wide:true,help:"Comma-separated styles used by Add suggested prompts: natural, studio, cinematic, expression."})
  ]);
  addRecipeGroup("Runtime and checkpoints","Memory, previews, and saved refinement checkpoints.",[
    objectField("Preview every","preview_every",config.preview_every??5,v=>set("preview_every",v),{type:"number"}),
    ...(isH3?[objectField("Saved preview quality","quality_preview_mode",config.quality_preview_mode||"one_frame",v=>set("quality_preview_mode",v),{type:"select",options:[{label:"Fast one-frame (recommended for frequent previews)",value:"one_frame"},{label:"Native five-frame + center frame (slower)",value:"five_frame"}],help:"Refinement updates always remain one-frame. Five-frame mode runs an additional no-gradient MiniMax inference whenever a preview is due; it improves decode detail but can substantially increase seconds per iteration."}),objectField("Five-frame preview steps","quality_preview_steps",config.quality_preview_steps??20,v=>set("quality_preview_steps",v),{type:"number",help:"Used only for optional five-frame quality previews, not for the refinement gradient."}),objectField("Always save final preview","quality_preview_final",config.quality_preview_final??true,v=>set("quality_preview_final",v),{type:"boolean",help:"Creates the selected preview at completion or early stop even when the final step is not on the normal cadence."})]:[]),
    objectField("Save every","save_every",config.save_every??10,v=>set("save_every",v),{type:"number"}),
    objectField("Blocks to swap","blocks_to_swap",config.blocks_to_swap??(isH3?35:10),v=>set("blocks_to_swap",v),{type:"number",help:isH3?"MiniMax H3 face refinement is substantially heavier than normal LoRA training. Start at 35 on 24 GB.":"Move inactive transformer blocks through CPU memory to reduce VRAM."}),
    objectField("GPU","gpu_id",config.gpu_id||"auto",v=>set("gpu_id",v)),
    objectField("Q/K/V/O only","qkvo_only",config.qkvo_only??true,v=>set("qkvo_only",v),{type:"boolean"}),
    objectField("Checkpoint VAE","checkpoint_vae",config.checkpoint_vae??true,v=>set("checkpoint_vae",v),{type:"boolean"})
  ],isH3);
  config.prompts=Array.isArray(config.prompts)?config.prompts:[];const promptHost=$("#face-prompts");promptHost.innerHTML=config.prompts.length?"":'<div class="empty">Add at least one refinement prompt.</div>';config.prompts.forEach((prompt,index)=>{const row=document.createElement("div");row.className="structured-item face-prompt-card";row.innerHTML=`<div class="structured-item-head"><div><strong>Prompt ${index+1}</strong><small>${esc(String(prompt).replace(/\s+/g," ").trim()||"Empty prompt")}</small></div><div class="structured-item-actions"><button>Remove</button></div></div><details class="prompt-details"><summary>Edit prompt</summary><div class="prompt-field-host"></div></details>`;row.querySelector(".prompt-field-host").append(objectField("Prompt","prompt",prompt,v=>{config.prompts[index]=v},{type:"textarea",wide:true}));row.querySelector(".structured-item-actions button").addEventListener("click",()=>{config.prompts.splice(index,1);renderFaceWorkspace();sync()});promptHost.append(row)});
  const poseHost=$("#pose-plan"),plan=config.pose_plan||{},buckets=plan.buckets||{};
  if(Object.keys(buckets).length){renderPosePlanTable(poseHost,config,buckets);renderPosePromptEditor($("#pose-prompt-editor"),config,buckets)}
  else {poseHost.innerHTML='<div class="empty">Enable pose-aware training to configure pose goals.</div>';$("#pose-prompt-editor").innerHTML=""}
  const evalHost=$("#face-eval-fields");evalHost.innerHTML=isH3?'<div class="issue warning">MiniMax H3 DRaFT refinement is available, but the fixed comparison recipe is not validated yet. Use standalone MiniMax H3 prompt previews to inspect checkpoints.</div>':"";if(!isH3)evalHost.append(objectField("Prompts per pose","evaluation_prompts_per_pose",config.evaluation_prompts_per_pose??1,v=>set("evaluation_prompts_per_pose",v),{type:"number"}),objectField("Seeds per prompt","evaluation_seeds_per_prompt",config.evaluation_seeds_per_prompt??2,v=>set("evaluation_seeds_per_prompt",v),{type:"number"}),objectField("Seed","evaluation_seed",config.evaluation_seed??42000,v=>set("evaluation_seed",v),{type:"number"}),objectField("Resolution","evaluation_resolution",config.evaluation_resolution??512,v=>set("evaluation_resolution",v),{type:"number"}),objectField("Steps","evaluation_steps",config.evaluation_steps??8,v=>set("evaluation_steps",v),{type:"number"}),objectField("LoRA strength","evaluation_lora_strength",config.evaluation_lora_strength??1,v=>set("evaluation_lora_strength",v),{type:"number"}));
  ["#face-baseline","#face-compare","#load-face-result","#open-face-results","#build-weak-pose-plan"].forEach(selector=>{$(selector).disabled=isH3});
  const report=config.preflight_report,excluded=new Set(config.excluded_reference_images||[]),poseNames=["uncertain","frontal","three_quarter_left","three_quarter_right","profile_left","profile_right","looking_up","looking_down"];
  const reportHost=$("#face-preflight-report");
  if(report){
    const allReferences=(report.scored_images||[]).map((item,index)=>({item,index}));
    const filter=state.faceReferenceFilter||"all";
    const filtered=allReferences.filter(({item})=>filter==="all"||filter==="flagged"&&(item.outlier||item.bucket==="uncertain")||filter==="excluded"&&excluded.has(item.path)||filter.startsWith("pose:")&&item.bucket===filter.slice(5));
    const pageSize=innerWidth<=680?8:innerWidth<=1000?12:24,pageCount=Math.max(1,Math.ceil(filtered.length/pageSize));
    state.faceReferencePage=Math.min(state.faceReferencePage,pageCount-1);
    const visible=filtered.slice(state.faceReferencePage*pageSize,(state.faceReferencePage+1)*pageSize);
    reportHost.innerHTML=`<div class="inspection-stats"><div><strong>${report.images_scanned||0}</strong><small>SCANNED</small></div><div><strong>${report.valid_faces||0}</strong><small>USABLE</small></div><div><strong>${allReferences.filter(x=>x.item.outlier).length}</strong><small>FLAGGED</small></div></div>${(report.warnings||[]).map(w=>`<div class="issue warning">${esc(w)}</div>`).join("")}<div class="reference-toolbar"><label for="reference-filter">Show</label><select id="reference-filter"><option value="all">All references</option><option value="flagged">Needs review</option><option value="excluded">Excluded</option>${poseNames.map(name=>`<option value="pose:${name}">${esc(name.replaceAll("_"," "))}</option>`).join("")}</select><span>${filtered.length?`${state.faceReferencePage*pageSize+1}–${Math.min((state.faceReferencePage+1)*pageSize,filtered.length)} of ${filtered.length}`:"No matches"}</span><div><button class="quiet" data-reference-page="-1" ${state.faceReferencePage===0?"disabled":""}>Previous</button><button class="quiet" data-reference-page="1" ${state.faceReferencePage>=pageCount-1?"disabled":""}>Next</button></div></div><div class="reference-grid">${visible.map(({item,index})=>{const filename=String(item.path||"").split(/[\\/]/).pop();return `<article class="reference-card ${excluded.has(item.path)?"excluded":""}" data-index="${index}"><img src="/api/face-image?path=${encodeURIComponent(item.path)}" loading="lazy" alt="${esc(filename)}"><strong>${item.outlier?"Review suggested":esc((item.bucket||"uncertain").replaceAll("_"," "))}</strong><small class="reference-filename" title="${esc(filename)}">${esc(filename)}</small><small>Identity ${Number(item.similarity||0).toFixed(3)} · confidence ${Number(item.confidence||0).toFixed(2)}</small><select class="reference-pose" aria-label="Correct detected pose">${poseNames.map(name=>`<option value="${name}" ${name===(item.bucket||"uncertain")?"selected":""}>${esc(name.replaceAll("_"," "))}</option>`).join("")}</select><label><input class="reference-use" type="checkbox" ${excluded.has(item.path)?"":"checked"}> Use reference</label></article>`}).join("")}</div>`;
    $("#reference-filter").value=filter;
    $("#reference-filter").addEventListener("change",event=>{state.faceReferenceFilter=event.target.value;state.faceReferencePage=0;renderFaceWorkspace()});
    $$("[data-reference-page]").forEach(button=>button.addEventListener("click",()=>{state.faceReferencePage+=Number(button.dataset.referencePage);renderFaceWorkspace();document.querySelector(".reference-toolbar")?.scrollIntoView({block:"start"})}));
  }else reportHost.innerHTML="No preflight report saved yet.";
  $$("#face-preflight-report .reference-card").forEach(card=>{
    const item=report.scored_images[Number(card.dataset.index)];
    card.querySelector(".reference-use").addEventListener("change",e=>{if(e.target.checked)excluded.delete(item.path);else excluded.add(item.path);config.excluded_reference_images=[...excluded];card.classList.toggle("excluded",!e.target.checked);sync()});
    card.querySelector(".reference-pose").addEventListener("change",e=>{item.bucket=e.target.value;item.confidence=1;item.pose_source="manual";report.pose_bucket_counts=Object.fromEntries(poseNames.map(name=>[name,(report.scored_images||[]).filter(entry=>entry.bucket===name&&!excluded.has(entry.path)).length]));renderFaceWorkspace();sync();toast("Pose correction saved in this workspace.")});
  });
  const faceStage=(state.settings.staged_training_config||[]).find(stage=>stage.type==="face_refinement");
  $("#face-stage-status").textContent=faceStage
    ? `Final face stage ready · ${faceStage.steps||config.steps||30} steps`
    : "Face refinement is not in the staged plan yet.";
}
function addOrUpdateFaceStage(){
  const config=faceConfig(),stages=Array.isArray(state.settings.staged_training_config)?state.settings.staged_training_config:[];
  const existing=stages.find(stage=>stage.type==="face_refinement");
  const stage={...existing,enabled:true,type:"face_refinement",label:existing?.label||"face-refinement",dataset_config:"",epochs:"",steps:String(config.steps??30),dop_mode:"inherit",depth_helpers_mode:"inherit"};
  state.settings.staged_training_config=stages.filter(item=>item.type!=="face_refinement");
  state.settings.staged_training_config.push(stage);
  state.settings.use_staged_training=true;
  renderPlan();renderFaceWorkspace();sync();
  toast(existing?"Face refinement updated as the final stage.":"Face refinement added as the final stage.");
}
function reviewFaceStage(){
  const hasStage=(state.settings.staged_training_config||[]).some(stage=>stage.type==="face_refinement");
  if(!hasStage)return toast("Add face refinement to the staged plan first.");
  go("plan");
  $$("[data-plan-tab]").find(button=>button.dataset.planTab==="stages")?.click();
}
function renderTools(){
  const host=$("#convert-fields");host.innerHTML="";
  host.append(
    objectField("LoRA to convert","convert_lora_path",state.settings.convert_lora_path||"",v=>state.settings.convert_lora_path=v,{wide:true,browseKind:"file",help:"The source .safetensors file is read only and is never overwritten."}),
    objectField("Output directory","convert_output_dir",state.settings.convert_output_dir||"",v=>state.settings.convert_output_dir=v,{wide:true,browseKind:"directory"}),
    objectField("Direction","convert_target",state.settings.convert_target||"default",v=>state.settings.convert_target=v,{type:"select",options:[{label:"Diffusers → Musubi / ComfyUI",value:"default"},{label:"Musubi / ComfyUI → Diffusers",value:"other"}]})
  );
}

function renderAllSettings() {
  const host = $("#settings-sections"); host.innerHTML = "";
  state.schema.sections.forEach(section => {
    const details = document.createElement("details"); details.className = "settings-group"; details.dataset.section = section.id;
    details.innerHTML = `<summary><span>${esc(section.title)}</span><small>${section.fields.length} options</small></summary><div class="guided-fields"></div>`;
    section.fields.forEach(field => details.querySelector(".guided-fields").append(fieldControl(field)));
    host.append(details);
  });
  filterSettings();
}
function filterSettings() {
  const query = ($("#setting-search").value || "").toLowerCase(), showAll = $("#show-all-modes").checked, mode = state.settings.training_mode;
  $$(".settings-group").forEach(group => {
    let visible = 0;
    group.querySelectorAll(".field").forEach(field => {
      const modes = field.dataset.modes ? field.dataset.modes.split("|") : [];
      const show = (!query || field.dataset.search.includes(query)) && (showAll || !modes.length || modes.includes(mode));
      field.style.display = show ? "" : "none"; if (show) visible++;
    });
    group.style.display = visible ? "" : "none"; if (query && visible) group.open = true;
  });
}

let datasetDraftTimer = null;
let datasetMediaTimer = null;
let datasetMediaRequest = 0;
const DATASET_HELP = {
  source_mode:"Folders pair media with sidecar caption files. JSONL manifests keep media paths and captions together in ordered records.",
  source:"This is passed to Musubi exactly as written. Relative paths resolve from the GUI's project folder, the same as training.",
  cache_directory:"Each source needs its own cache folder. Reusing a cache between sources can mix incompatible latent or text caches.",
  resolution:"A single value means a square target. Two values mean width and height. Leave the source override off to inherit [general].",
  num_repeats:"Repeats multiply this source's influence per epoch. The overview shows the resulting effective sample count.",
  batch_size:"A per-source batch override. Larger batches need more VRAM; leave it inherited unless this source needs different packing.",
  caption_extension:"Sidecar suffix such as .txt or .caption.txt. For image folders, missing sidecars are excluded; videos fail when a caption is missing.",
  enable_bucket:"Groups different aspect ratios efficiently. Inherit keeps this source linked to the document default.",
  bucket_no_upscale:"Prevents small media from being enlarged to a bucket, preserving detail at the cost of variable effective sizes.",
  control_directory:"Optional matching controls. Musubi pairs files by stem, so names should match the training media.",
  target_frames:"Video sequence lengths Musubi will cache and train, for example 1, 25, 49, 81.",
  frame_extraction:"Controls where clips come from inside each source video.",
  multiple_target:"Enables numbered image targets used by layered-image trainers.",
  no_resize_control:"Keeps control media at its original size instead of matching the training target.",
};
const datasetValueText=value=>Array.isArray(value)?value.join(", "):(value??"");
const datasetBasename=value=>String(value||"").replaceAll("\\","/").split("/").filter(Boolean).pop()||"Untitled source";
const formatBytes=value=>{const n=Number(value||0);if(n<1024)return `${n} B`;if(n<1024**2)return `${(n/1024).toFixed(1)} KB`;if(n<1024**3)return `${(n/1024**2).toFixed(1)} MB`;return `${(n/1024**3).toFixed(1)} GB`};

async function loadDatasetDocument({quiet = false} = {}) {
  const path = $("#dataset-path").value.trim();
  if (!path) {
    if (!quiet) toast("Choose a dataset TOML first.");
    return false;
  }
  if(state.dataset&&(state.datasetDirty||state.datasetFormDirty||state.datasetRawDirty)){
    const currentPath=state.dataset.path||state.settings.dataset_config||"";
    if(quiet){
      $("#dataset-path").value=currentPath;
      toast("The current TOML has unsaved changes. Save it before loading another dataset.", "error");
      return false;
    }
    if(!confirm("Discard the unsaved TOML changes and reload from disk?")){
      $("#dataset-path").value=currentPath;
      return false;
    }
  }
  const button = $("#load-dataset");
  try {
    await withBusy(button, "Loading…", async () => {
      const payload = await api(`/api/dataset?path=${encodeURIComponent(path)}`);
      state.datasetFormDirty=false;state.datasetRawDirty=false;state.datasetInventories={};state.datasetAudit=null;state.datasetMedia=null;state.samplingEstimate=null;
      state.selectedDataset=payload.datasets.length?0:-1;state.datasetTab=payload.datasets.length?"media":"settings";
      renderDataset(payload,state.selectedDataset);
      const linkedPath=payload.path||path,recipeChanged=!sameLocalPath(state.settings.dataset_config,linkedPath);
      state.settings.dataset_config=linkedPath;
      setDatasetDirty(false);
      sync(recipeChanged);
    });
    if(state.selectedDataset>=0)loadDatasetMedia({skipFlush:true}).catch(error=>renderDatasetMediaError(error));
    return true;
  } catch (error) {
    renderIssues([{level:"error", message:error.message}]);
    if (!quiet) toast(error.message, "error");
    return false;
  }
}
function ensureDatasetLoaded() {
  const path = ($("#dataset-path").value || state.settings.dataset_config || "").trim();
  if (!path || (state.dataset && sameLocalPath(state.dataset.path,path))) return;
  $("#dataset-path").value = path;
  loadDatasetDocument({quiet:true});
}
async function loadDatasetForSettings({quiet = false} = {}) {
  const path = String(state.settings.dataset_config || "").trim();
  if (!path) {
    if (!quiet) toast("This recipe has no Dataset TOML. Choose one before starting.", "error");
    return false;
  }
  if (state.dataset && sameLocalPath(state.dataset.path, path)) return true;
  $("#dataset-path").value = path;
  return loadDatasetDocument({quiet});
}
function renderDataset(payload, selected = state.selectedDataset) {
  state.dataset=payload;
  if(selected===-1)state.selectedDataset=-1;
  else state.selectedDataset=payload.datasets.length?Math.max(0,Math.min(Number(selected)||0,payload.datasets.length-1)):-1;
  $("#dataset-path").value=payload.path||$("#dataset-path").value;
  $("#dataset-source").value=payload.text||"";
  $("#dataset-count").textContent=payload.datasets.length;
  $("#inspect-dataset").disabled=!payload.datasets.length;
  $("#add-image-dataset").disabled=false;$("#add-video-dataset").disabled=false;
  $("#dataset-welcome").style.display=payload?"none":"";
  $("#dataset-content").hidden=!payload;
  $("#toml-preservation-state").textContent=payload.preservation_available?"Comments and structure preserved":"Formatting preservation unavailable";
  renderDatasetRail();
  renderDatasetHead();
  renderDatasetEditor();
  renderDatasetOverview();
  renderIssues(payload.issues||[]);
  setDatasetTab(state.selectedDataset<0&&state.datasetTab==="media"?"settings":state.datasetTab,{load:false});
  setDatasetDirty(state.datasetDirty);
}
function renderDatasetRail(){
  const payload=state.dataset,host=$("#dataset-list"),defaults=$("#dataset-defaults");
  defaults.classList.toggle("active",state.selectedDataset===-1);
  defaults.setAttribute("aria-current",state.selectedDataset===-1?"true":"false");
  if(!payload){host.className="dataset-list empty";host.textContent="Load a TOML to begin.";return}
  const knownInventories=state.datasetAudit?.datasets||Object.values(state.datasetInventories),knownEffective=knownInventories.reduce((sum,item)=>sum+Number(item.effective_samples||0),0);
  host.classList.toggle("empty",!payload.datasets.length);
  host.innerHTML=payload.datasets.length?payload.datasets.map((dataset,index)=>{
    const inventory=state.datasetInventories[index]||state.datasetAudit?.datasets?.find(item=>item.index===index);
    const sourceIssues=(payload.issues||[]).filter(issue=>issue.dataset_index===index);
    const weight=inventory&&knownEffective?Math.round(Number(inventory.effective_samples||0)/knownEffective*100):0;
    const effective=inventory?`${inventory.trainer_usable_count} usable · ×${inventory.repeats||dataset.repeats} = ${inventory.effective_samples}${knownInventories.length===payload.datasets.length?` · ${weight}%`:""}`:`${dataset.source_mode==="jsonl"?"JSONL manifest":"Folder"} · ×${dataset.repeats}`;
    const warning=sourceIssues.some(issue=>issue.level==="error")?"error":sourceIssues.length?"warning":inventory?.missing_caption_count?"warning":"";
    return `<button class="dataset-source ${index===state.selectedDataset?"active":""}" data-index="${index}" aria-current="${index===state.selectedDataset?"true":"false"}"><span class="source-icon">${dataset.kind==="video"?"▶":"▧"}</span><span><strong>${esc(datasetBasename(dataset.source)||`Source ${index+1}`)}</strong><small>Source ${index+1} · ${esc(effective)}</small></span>${warning?`<i class="${warning}" title="${sourceIssues.length||inventory.missing_caption_count} item(s) need attention"></i>`:""}${inventory&&knownInventories.length===payload.datasets.length?`<b class="source-weight" style="width:${weight}%"></b>`:""}</button>`;
  }).join(""):"No sources in this document.";
  host.querySelectorAll("[data-index]").forEach(button=>button.addEventListener("click",async()=>{
    const index=Number(button.dataset.index);
    try{
      await flushDatasetDraft();
      state.selectedDataset=index;state.datasetTab="media";state.datasetMediaPage=1;state.datasetMedia=null;
      renderDataset(state.dataset,index);
      await loadDatasetMedia({skipFlush:true});
      $(`#dataset-list [data-index="${index}"]`)?.focus();
    }catch(error){toast(error.message,"error")}
  }));
}
function renderDatasetHead(){
  const host=$("#dataset-source-head"),dataset=state.dataset?.datasets?.[state.selectedDataset];
  if(state.selectedDataset<0){
    host.innerHTML=`<div class="dataset-form-head"><div><p class="kicker">DOCUMENT DEFAULTS</p><h2>Shared source behavior</h2><small>Sources inherit these values until you enable a source-level override.</small></div></div>`;
    return;
  }
  if(!dataset){host.innerHTML="";return}
  const advanced=dataset.advanced_keys?.length?` · ${dataset.advanced_keys.length} advanced key${dataset.advanced_keys.length===1?"":"s"} preserved`:"";
  host.innerHTML=`<div class="dataset-form-head"><div><p class="kicker">${esc(dataset.kind)} · ${esc(dataset.source_mode==="jsonl"?"JSONL MANIFEST":"MEDIA FOLDER")}</p><h2>${esc(datasetBasename(dataset.source))}</h2><small>${esc(dataset.resolved_source||dataset.source)}${esc(advanced)}</small></div><div class="dataset-actions"><button data-source-action="up" title="Move source earlier" ${dataset.index===0?"disabled":""}>↑</button><button data-source-action="down" title="Move source later" ${dataset.index===state.dataset.datasets.length-1?"disabled":""}>↓</button><button data-source-action="duplicate">Copy</button><button data-source-action="remove">Remove</button></div></div>`;
  host.querySelector('[data-source-action="up"]')?.addEventListener("click",()=>moveDatasetSource(-1));
  host.querySelector('[data-source-action="down"]')?.addEventListener("click",()=>moveDatasetSource(1));
  host.querySelector('[data-source-action="duplicate"]')?.addEventListener("click",()=>mutateDataset("/api/dataset/duplicate",{index:dataset.index},dataset.index+1,{message:"Source copied. Choose a unique cache directory before training."}));
  host.querySelector('[data-source-action="remove"]')?.addEventListener("click",()=>{
    if(state.dataset.datasets.length===1)return toast("A dataset TOML needs at least one source.","error");
    if(confirm(`Remove Source ${dataset.index+1} from this TOML draft?\n\nMedia and caption files will not be deleted.`))mutateDataset("/api/dataset/remove",{index:dataset.index},Math.max(0,dataset.index-1));
  });
}
async function selectDatasetDefaults(){
  try{
    await flushDatasetDraft();state.selectedDataset=-1;state.datasetTab="settings";state.datasetMedia=null;renderDataset(state.dataset,-1);$("#dataset-defaults").focus();
  }catch(error){toast(error.message,"error")}
}
function datasetField({key,label,value="",wide=false,type="text",tip="",placeholder="",browse=false,options=null,small=""}){
  const id=`dataset-${key.replaceAll("_","-")}-${++controlSequence}`,help=tip||DATASET_HELP[key]||`${label} is written directly into the selected TOML table.`;
  const control=options?`<select id="${id}" data-key="${esc(key)}">${options.map(([optionValue,copy])=>`<option value="${esc(optionValue)}" ${String(value)===String(optionValue)?"selected":""}>${esc(copy)}</option>`).join("")}</select>`:`<input id="${id}" type="${type}" data-key="${esc(key)}" value="${esc(value)}" placeholder="${esc(placeholder)}">`;
  return `<div class="field${wide?" wide":""}"><label class="field-label" for="${id}" data-tip="${esc(help)}">${esc(label)}</label><div class="${browse?"path-control":""}">${control}${browse?`<button type="button" class="quiet" data-browse-key="${esc(key)}">Browse</button>`:""}</div>${small?`<small>${esc(small)}</small>`:""}</div>`;
}
function inheritedDatasetField(dataset,key,label,{type="text",wide=false,placeholder=""}={}){
  const origin=dataset.value_origins?.[key]||"default",raw=dataset.raw_values?.[key],effective=dataset.effective_values?.[key];
  const id=`dataset-${key.replaceAll("_","-")}-${++controlSequence}`,checked=origin==="dataset",help=DATASET_HELP[key]||label;
  const originCopy=origin==="general"?"Inherited from [general]":`Using Musubi default`;
  return `<div class="field${wide?" wide":""} inherited-field"><div class="field-label-row"><label class="field-label" for="${id}" data-tip="${esc(help)}">${esc(label)}</label><label class="override-toggle"><input type="checkbox" data-override="${esc(key)}" ${checked?"checked":""}> Override for source</label></div><input id="${id}" type="${type}" data-key="${esc(key)}" value="${esc(datasetValueText(checked?raw:effective))}" placeholder="${esc(placeholder)}" ${checked?"":"disabled"}><small data-origin-copy>${esc(checked?"Source override":`${originCopy}: ${datasetValueText(effective)??"—"}`)}</small></div>`;
}
function renderDatasetEditor(){
  const host=$("#dataset-editor"),dataset=state.dataset?.datasets?.[state.selectedDataset];
  host.innerHTML="";
  if(state.selectedDataset<0){
    const general=state.dataset?.general||{};
    host.innerHTML=`<div class="setting-section"><div class="setting-section-head"><div><h3>Training defaults</h3><p>Blank fields keep Musubi’s built-in behavior. Sources can override any inherited value explicitly.</p></div></div><div class="guided-fields">
      ${datasetField({key:"resolution",label:"Default resolution",value:datasetValueText(general.resolution),placeholder:"960, 544",small:"One value for square, or width and height."})}
      ${datasetField({key:"num_repeats",label:"Default repeats",value:general.num_repeats??"",type:"number",placeholder:"1"})}
      ${datasetField({key:"batch_size",label:"Default batch size",value:general.batch_size??"",type:"number",placeholder:"1"})}
      ${datasetField({key:"caption_extension",label:"Caption extension",value:general.caption_extension??"",placeholder:".txt"})}
      ${datasetField({key:"enable_bucket",label:"Aspect-ratio buckets",value:general.enable_bucket==null?"":String(general.enable_bucket),options:[["","Musubi default (off)"],["true","Enabled"],["false","Disabled"]]})}
      ${datasetField({key:"bucket_no_upscale",label:"Prevent bucket upscaling",value:general.bucket_no_upscale==null?"":String(general.bucket_no_upscale),options:[["","Musubi default (off)"],["true","Enabled"],["false","Disabled"]]})}
      <button class="primary dataset-apply" data-apply-dataset>Apply defaults to TOML draft</button>
    </div></div>`;
    bindDatasetEditorControls();
    return;
  }
  if(!dataset)return;
  const raw=dataset.raw_values||{},kind=dataset.kind,mode=dataset.source_mode;
  const boolOptions=[["","Use trainer default"],["true","Enabled"],["false","Disabled"]];
  host.innerHTML=`<div class="setting-section"><div class="setting-section-head"><div><h3>Media source</h3><p>Choose a normal folder with sidecar captions or an ordered JSONL manifest.</p></div></div><div class="guided-fields">
    ${datasetField({key:"_source_mode",label:"Source format",value:mode,options:[["directory","Media folder"],["jsonl","JSONL manifest"]]})}
    ${datasetField({key:"_source_path",label:mode==="jsonl"?"Manifest file":`${kind==="video"?"Video":"Image"} folder`,value:dataset.source,wide:true,browse:true,tip:DATASET_HELP.source})}
    ${datasetField({key:"cache_directory",label:"Cache directory",value:raw.cache_directory||"",wide:true,browse:true})}
  </div></div>
  <div class="setting-section"><div class="setting-section-head"><div><h3>Training balance</h3><p>Inherited values remain linked to Document defaults until you enable an override.</p></div></div><div class="guided-fields">
    ${inheritedDatasetField(dataset,"resolution","Training resolution")}
    ${inheritedDatasetField(dataset,"num_repeats","Repeats",{type:"number"})}
    ${inheritedDatasetField(dataset,"batch_size","Batch size",{type:"number"})}
    ${inheritedDatasetField(dataset,"caption_extension","Caption extension",{placeholder:".txt"})}
    ${datasetField({key:"enable_bucket",label:"Aspect-ratio buckets",value:raw.enable_bucket==null?"":String(raw.enable_bucket),options:[["","Inherit document default"],["true","Enabled for this source"],["false","Disabled for this source"]]})}
    ${datasetField({key:"bucket_no_upscale",label:"Prevent bucket upscaling",value:raw.bucket_no_upscale==null?"":String(raw.bucket_no_upscale),options:[["","Inherit document default"],["true","Enabled for this source"],["false","Disabled for this source"]]})}
  </div></div>
  <div class="setting-section"><div class="setting-section-head"><div><h3>${kind==="video"?"Sequence extraction":"Controls and layered targets"}</h3><p>${kind==="video"?"Choose which temporal windows Musubi takes from each video.":"Optional fields for control/reference and layered-image training."}</p></div></div><div class="guided-fields">
    ${datasetField({key:"control_directory",label:"Control directory",value:raw.control_directory||"",wide:true,browse:true})}
    ${kind==="video"?`
      ${datasetField({key:"target_frames",label:"Target frames",value:datasetValueText(raw.target_frames),wide:true,placeholder:"1, 25, 49"})}
      ${datasetField({key:"frame_extraction",label:"Frame extraction",value:raw.frame_extraction||"",options:[["","Use trainer default (head)"],["head","Head"],["chunk","Chunks"],["slide","Sliding windows"],["uniform","Uniform samples"],["full","Full sequence"]]})}
      ${datasetField({key:"frame_stride",label:"Frame stride",value:raw.frame_stride??"",type:"number"})}
      ${datasetField({key:"frame_sample",label:"Uniform samples",value:raw.frame_sample??"",type:"number"})}
      ${datasetField({key:"max_frames",label:"Maximum frames",value:raw.max_frames??"",type:"number"})}
      ${datasetField({key:"source_fps",label:"Source FPS",value:raw.source_fps??"",type:"number"})}
    `:`
      ${datasetField({key:"multiple_target",label:"Multiple targets",value:raw.multiple_target==null?"":String(raw.multiple_target),options:boolOptions})}
      ${datasetField({key:"no_resize_control",label:"Keep control size",value:raw.no_resize_control==null?"":String(raw.no_resize_control),options:boolOptions})}
      ${datasetField({key:"control_resolution",label:"Control resolution",value:datasetValueText(raw.control_resolution),placeholder:"width, height"})}
    `}
  </div></div>
  ${dataset.advanced_keys?.length?`<details class="preserved-fields"><summary>${dataset.advanced_keys.length} advanced field${dataset.advanced_keys.length===1?"":"s"} preserved in TOML</summary><p>${dataset.advanced_keys.map(esc).join(" · ")}</p></details>`:""}
  <div class="dataset-settings-footer"><span>Visual changes update the canonical TOML draft automatically.</span><button class="primary" data-apply-dataset>Apply now</button></div>`;
  bindDatasetEditorControls();
}
function bindDatasetEditorControls(){
  const host=$("#dataset-editor");
  host.querySelectorAll("[data-override]").forEach(toggle=>toggle.addEventListener("change",()=>{
    const input=host.querySelector(`[data-key="${CSS.escape(toggle.dataset.override)}"]`);if(input)input.disabled=!toggle.checked;markDatasetFormChanged();if(toggle.checked)input?.focus();
  }));
  host.querySelectorAll("[data-key]").forEach(input=>input.addEventListener(input.tagName==="SELECT"?"change":"input",markDatasetFormChanged));
  host.querySelectorAll("[data-browse-key]").forEach(button=>button.addEventListener("click",()=>withBusy(button,"Choosing…",async()=>{
    const key=button.dataset.browseKey,input=host.querySelector(`[data-key="${CSS.escape(key)}"]`),mode=host.querySelector('[data-key="_source_mode"]')?.value;
    const kind=key==="_source_path"&&mode==="jsonl"?"file":"directory";
    const result=await api("/api/path/select",{method:"POST",body:JSON.stringify({kind,initial:input.value})});
    if(result.path){input.value=result.path;markDatasetFormChanged()}
  }).catch(error=>toast(error.message,"error"))));
  host.querySelector('[data-key="_source_mode"]')?.addEventListener("change",event=>{
    const pathLabel=host.querySelector('[data-key="_source_path"]')?.closest(".field")?.querySelector(".field-label");
    if(pathLabel)pathLabel.textContent=event.target.value==="jsonl"?"Manifest file":`${state.dataset.datasets[state.selectedDataset].kind==="video"?"Video":"Image"} folder`;
  });
  host.querySelector("[data-apply-dataset]")?.addEventListener("click",event=>withBusy(event.currentTarget,"Applying…",()=>persistDatasetFormDraft({announce:true,rerender:true})).catch(error=>toast(error.message,"error")));
}
function markDatasetFormChanged(){
  state.datasetFormDirty=true;setDatasetDirty(true);clearTimeout(datasetDraftTimer);
  datasetDraftTimer=setTimeout(()=>persistDatasetFormDraft().catch(error=>{renderIssues([{level:"error",message:error.message}]);$("#dataset-document-state").textContent="Fix source settings"}),800);
}
function parseDatasetNumber(value,label,{integer=true,optional=true}={}){
  const text=String(value??"").trim();if(!text&&optional)return "";
  const parsed=Number(text);if(!Number.isFinite(parsed)||(integer&&!Number.isInteger(parsed))||parsed<=0)throw new Error(`${label} must be a positive ${integer?"whole number":"number"}.`);
  return parsed;
}
function parseDatasetDimensions(value,label,{allowScalar=true,optional=true}={}){
  const text=String(value??"").trim();if(!text&&optional)return "";
  const parts=text.split(/[,;x×\s]+/).filter(Boolean).map(piece=>parseDatasetNumber(piece,label,{integer:true,optional:false}));
  if(parts.length===1&&allowScalar)return parts[0];
  if(parts.length!==2)throw new Error(`${label} needs ${allowScalar?"one value or ":""}width and height.`);
  return parts;
}
function parseDatasetList(value,label){
  const text=String(value??"").trim();if(!text)return "";
  return text.split(/[,;\s]+/).filter(Boolean).map(piece=>parseDatasetNumber(piece,label,{integer:true,optional:false}));
}
function parseDatasetBoolean(value){return value===""?"":value==="true"}
function collectDatasetEditorChanges(){
  const host=$("#dataset-editor"),changes={};
  if(state.selectedDataset<0){
    host.querySelectorAll("[data-key]").forEach(input=>{
      const key=input.dataset.key,value=input.value;
      if(key==="resolution")changes[key]=parseDatasetDimensions(value,"Default resolution");
      else if(["num_repeats","batch_size"].includes(key))changes[key]=parseDatasetNumber(value,key==="num_repeats"?"Default repeats":"Default batch size");
      else if(["enable_bucket","bucket_no_upscale"].includes(key))changes[key]=parseDatasetBoolean(value);
      else changes[key]=value.trim();
    });
    return changes;
  }
  const dataset=state.dataset.datasets[state.selectedDataset],mode=host.querySelector('[data-key="_source_mode"]').value,path=host.querySelector('[data-key="_source_path"]').value.trim();
  const directoryKey=`${dataset.kind}_directory`,jsonlKey=`${dataset.kind}_jsonl_file`;
  changes[mode==="jsonl"?jsonlKey:directoryKey]=path;changes[mode==="jsonl"?directoryKey:jsonlKey]="";
  host.querySelectorAll("[data-key]").forEach(input=>{
    const key=input.dataset.key;if(key.startsWith("_"))return;
    const override=host.querySelector(`[data-override="${CSS.escape(key)}"]`);
    if(override&&!override.checked){changes[key]="";return}
    const value=input.value;
    if(["resolution","control_resolution"].includes(key))changes[key]=parseDatasetDimensions(value,key==="resolution"?"Training resolution":"Control resolution");
    else if(key==="target_frames")changes[key]=parseDatasetList(value,"Target frames");
    else if(["num_repeats","batch_size","frame_stride","frame_sample","max_frames","fp_latent_window_size"].includes(key))changes[key]=parseDatasetNumber(value,key.replaceAll("_"," "));
    else if(key==="source_fps")changes[key]=parseDatasetNumber(value,"Source FPS",{integer:false});
    else if(["enable_bucket","bucket_no_upscale","multiple_target","no_resize_control"].includes(key))changes[key]=parseDatasetBoolean(value);
    else changes[key]=value.trim();
  });
  return changes;
}
async function persistDatasetFormDraft({announce=false,rerender=false}={}){
  if(!state.datasetFormDirty&&!announce)return state.dataset;
  clearTimeout(datasetDraftTimer);
  const selected=state.selectedDataset,changes=collectDatasetEditorChanges(),endpoint=selected<0?"/api/dataset/general":"/api/dataset/update",beforeText=$("#dataset-source").value,wasDirty=state.datasetDirty;
  const body={text:$("#dataset-source").value,path:$("#dataset-path").value,changes};
  if(selected>=0)body.index=state.dataset.datasets[selected].index;
  const payload=await api(endpoint,{method:"POST",body:JSON.stringify(body)});
  state.dataset=payload;state.datasetFormDirty=false;state.datasetRawDirty=false;$("#dataset-source").value=payload.text||"";
  const changed=beforeText!==(payload.text||"");setDatasetDirty(wasDirty||changed);renderDatasetRail();renderDatasetHead();renderDatasetOverview();renderIssues(payload.issues||[]);
  if(rerender)renderDataset(payload,selected);
  if(announce)toast(changed?"Settings applied to the TOML draft. Save when ready.":"No TOML changes were needed.");
  return payload;
}
async function parseDatasetRawDraft({announce=false}={}){
  if(!state.datasetRawDirty)return state.dataset;
  const payload=await api("/api/dataset/parse",{method:"POST",body:JSON.stringify({path:$("#dataset-path").value,text:$("#dataset-source").value})});
  state.datasetRawDirty=false;state.datasetFormDirty=false;state.dataset=payload;
  renderDataset(payload,state.selectedDataset);setDatasetDirty(true);
  if(announce)toast("TOML changes applied to the visual workspace.");
  return payload;
}
async function flushDatasetDraft(){
  clearTimeout(datasetDraftTimer);
  if(state.datasetFormDirty)await persistDatasetFormDraft();
  if(state.datasetRawDirty)await parseDatasetRawDraft();
  return state.dataset;
}
async function saveDatasetDocument({announce=true}={}){
  await flushDatasetDraft();
  const path=$("#dataset-path").value,expected=sameLocalPath(state.dataset?.path,path)?state.dataset.disk_revision:null;
  const payload=await api("/api/dataset/save",{method:"POST",body:JSON.stringify({path,text:$("#dataset-source").value,expected_revision:expected})});
  state.datasetFormDirty=false;state.datasetRawDirty=false;renderDataset(payload,state.selectedDataset);setDatasetDirty(false);
  const recipeChanged=!sameLocalPath(state.settings.dataset_config,payload.path);
  state.settings.dataset_config=payload.path;sync(recipeChanged);
  if(announce)toast("Dataset TOML saved and linked to this recipe.");
  return payload;
}
async function mutateDataset(endpoint,extra,selected=state.selectedDataset,{message="Dataset document updated. Save when ready."}={}){
  try{
    await flushDatasetDraft();
    const payload=await api(endpoint,{method:"POST",body:JSON.stringify({text:$("#dataset-source").value,path:$("#dataset-path").value,...extra})});
    state.datasetInventories={};state.datasetAudit=null;state.datasetMedia=null;
    renderDataset(payload,selected);setDatasetDirty(true);toast(message);
    if(selected>=0&&state.datasetTab==="media")loadDatasetMedia({skipFlush:true}).catch(error=>renderDatasetMediaError(error));
  }catch(error){renderIssues([{level:"error",message:error.message}]);toast(error.message,"error")}
}
async function moveDatasetSource(direction){
  const index=state.selectedDataset,destination=index+direction;if(index<0||destination<0||destination>=state.dataset.datasets.length)return;
  await mutateDataset("/api/dataset/move",{index,destination},destination,{message:`Source moved to position ${destination+1}.`});
}
function setDatasetTab(tab,{load=true}={}){
  if(state.selectedDataset<0&&["media","health"].includes(tab))tab="settings";
  state.datasetTab=tab;
  $$("[data-dataset-tab]").forEach(button=>{const active=button.dataset.datasetTab===tab;button.classList.toggle("active",active);button.setAttribute("aria-selected",String(active));button.tabIndex=active?0:-1;button.disabled=state.selectedDataset<0&&["media","health"].includes(button.dataset.datasetTab)});
  $$("[data-dataset-pane]").forEach(pane=>pane.classList.toggle("active",pane.dataset.datasetPane===tab));
  if(load&&tab==="media"&&state.selectedDataset>=0)loadDatasetMedia({page:state.datasetMediaPage}).catch(error=>renderDatasetMediaError(error));
}
function renderDatasetMediaError(error){
  $("#dataset-media-summary").innerHTML="";
  $("#dataset-media-grid").className="dataset-media-grid empty";
  $("#dataset-media-grid").innerHTML=`<div class="media-empty"><span>!</span><h3>Could not read this source</h3><p>${esc(error.message)}</p><button class="quiet" data-open-source-settings>Review source settings</button></div>`;
  $("#dataset-media-grid").querySelector("[data-open-source-settings]")?.addEventListener("click",()=>setDatasetTab("settings"));
  $("#dataset-media-pager").innerHTML="";
}
async function loadDatasetMedia({page=state.datasetMediaPage||1,skipFlush=false}={}){
  if(state.selectedDataset<0||!state.dataset)return;
  if(!skipFlush)await flushDatasetDraft();
  const request=++datasetMediaRequest;state.datasetMediaPage=page;
  $("#dataset-media-grid").className="dataset-media-grid loading";
  $("#dataset-media-grid").innerHTML=`${Array.from({length:8},()=>'<div class="media-skeleton"></div>').join("")}`;
  const payload=await api("/api/dataset/media",{method:"POST",body:JSON.stringify({text:$("#dataset-source").value,path:$("#dataset-path").value,index:state.selectedDataset,page,page_size:24,query:state.datasetMediaQuery,filter:state.datasetMediaFilter})});
  if(request!==datasetMediaRequest)return;
  state.datasetMedia=payload;state.datasetMediaPage=payload.page;state.datasetInventories[state.selectedDataset]=payload.overview;
  renderDatasetMedia();renderDatasetRail();renderDatasetOverview();
}
function renderDatasetMedia(){
  const payload=state.datasetMedia,summary=$("#dataset-media-summary"),grid=$("#dataset-media-grid"),pager=$("#dataset-media-pager");
  if(!payload){renderDatasetMediaError(new Error("Select a source to browse its media."));return}
  const overview=payload.overview,captionBase=overview.primary_count??overview.media_count,coverage=captionBase?Math.round(overview.caption_count/captionBase*100):0;
  const missingCopy=payload.source.kind==="image"?"Missing captions are excluded from image training.":"Missing video captions will stop dataset loading.";
  summary.innerHTML=`<div class="media-health-strip"><div><strong>${overview.trainer_usable_count}</strong><small>TRAINER-USABLE</small></div><div><strong>${coverage}%</strong><small>CAPTION COVERAGE</small></div><div><strong>${overview.effective_samples}</strong><small>AFTER ×${overview.repeats} REPEATS</small></div><div class="${overview.missing_caption_count?"warning":""}"><strong>${overview.missing_caption_count}</strong><small>MISSING CAPTIONS</small></div></div>${overview.missing_caption_count?`<p class="media-health-note">${esc(missingCopy)}</p>`:""}`;
  grid.className=`dataset-media-grid${payload.items.length?"":" empty"}`;
  const stateCopy={eligible:"Ready",warning:"Review",excluded:"Excluded",error:"Error",paired_target:"Layer target"};
  grid.innerHTML=payload.items.length?payload.items.map((item,index)=>{
    const preview=!item.token?`<div class="media-placeholder"><span>${item.missing_media?"!":item.kind==="video"?"▶":"▧"}</span></div>`:item.preview_kind==="video"?`<video src="/api/dataset/media-file?token=${encodeURIComponent(item.token)}" preload="metadata" muted playsinline></video>`:`<img src="/api/dataset/media-file?token=${encodeURIComponent(item.token)}" loading="lazy" alt="">`;
    const dimensions=item.width&&item.height?`${item.width}×${item.height}`:item.kind==="video"?"Video":formatBytes(item.bytes);
    const caption=item.caption?.trim()||({missing:"No caption file",empty:"Caption is empty",unreadable:"Caption is not UTF-8",not_configured:"No caption extension"}[item.caption_state]||"No caption");
    return `<button class="dataset-media-card ${esc(item.training_state)}" data-media-index="${index}"><div class="media-thumb">${preview}<span class="media-state">${esc(stateCopy[item.training_state]||item.training_state)}</span>${item.controls?.length?`<span class="control-badge">${item.controls.length} control${item.controls.length===1?"":"s"}</span>`:""}</div><div class="media-card-copy"><strong title="${esc(item.name)}">${esc(item.name)}</strong><small>${esc(dimensions)}${item.target_count?` · ${item.target_count} target${item.target_count===1?"":"s"}`:""}</small><p>${esc(caption)}</p></div></button>`;
  }).join(""):`<div class="media-empty"><span>⌕</span><h3>No matching media</h3><p>Try another filter or search phrase.</p></div>`;
  grid.querySelectorAll("[data-media-index]").forEach(button=>button.addEventListener("click",()=>openDatasetMedia(Number(button.dataset.mediaIndex))));
  pager.innerHTML=payload.pages>1?`<button class="quiet" data-media-page="${payload.page-1}" ${payload.page<=1?"disabled":""}>← Previous</button><span>Page ${payload.page} of ${payload.pages} · ${payload.total} matches</span><button class="quiet" data-media-page="${payload.page+1}" ${payload.page>=payload.pages?"disabled":""}>Next →</button>`:`<span>${payload.total} item${payload.total===1?"":"s"}</span>`;
  pager.querySelectorAll("[data-media-page]").forEach(button=>button.addEventListener("click",()=>loadDatasetMedia({page:Number(button.dataset.mediaPage)}).catch(error=>renderDatasetMediaError(error))));
}
function renderDatasetOverview(){
  const host=$("#dataset-overview"),payload=state.dataset;
  if(!payload){return}
  let reports=state.datasetAudit?.datasets||Object.entries(state.datasetInventories).map(([index,overview])=>({index:Number(index),...overview}));
  const media=reports.reduce((sum,item)=>sum+Number(item.media_count||0),0),primaries=reports.reduce((sum,item)=>sum+Number(item.primary_count??item.media_count??0),0),captions=reports.reduce((sum,item)=>sum+Number(item.caption_count||0),0),effective=reports.reduce((sum,item)=>sum+Number(item.effective_samples||0),0);
  const scanned=reports.length,coverage=primaries?`${Math.round(captions/primaries*100)}%`:scanned?"0%":"—";
  const issueCount=(payload.issues||[]).length+reports.reduce((sum,item)=>sum+Number(item.missing_caption_count||0)+Number(item.unreadable_count||0)+(item.error?1:0),0);
  host.innerHTML=`<div><small>SOURCES</small><strong>${payload.datasets.length}</strong></div><div><small>MEDIA${scanned<payload.datasets.length?" SCANNED":""}</small><strong>${scanned?media:"—"}</strong></div><div><small>CAPTION COVERAGE</small><strong>${coverage}</strong></div><div><small>EFFECTIVE SAMPLES</small><strong>${scanned?effective:"—"}</strong></div><div class="${issueCount?"warning":"ok"}"><small>HEALTH</small><strong>${issueCount?`${issueCount} to review`:scanned===payload.datasets.length?"Ready":"Audit pending"}</strong></div>`;
}
function renderIssues(issues,target="#dataset-issues"){
  const host=$(target);if(!host)return;
  host.innerHTML=issues.length?issues.map(issue=>`<div class="issue ${esc(issue.level)}">${issue.dataset_index==null?"":`Source ${issue.dataset_index+1}: `}${esc(issue.message)}</div>`).join(""):`<div class="issue ok">TOML structure looks valid.</div>`;
}
async function inspectDataset(){
  await flushDatasetDraft();
  const payload=await api("/api/dataset/inspect",{method:"POST",body:JSON.stringify({text:$("#dataset-source").value,path:$("#dataset-path").value})});
  state.datasetAudit=payload;
  payload.datasets.forEach(report=>{if(!report.error)state.datasetInventories[report.index]=report});
  const host=$("#dataset-inspection"),auditEffective=payload.datasets.reduce((sum,item)=>sum+Number(item.effective_samples||0),0);
  host.innerHTML=`<div class="audit-intro"><div><p class="kicker">TRAINER-PARITY AUDIT</p><h3>${payload.datasets.length} source${payload.datasets.length===1?"":"s"} checked</h3><p>Counts follow Musubi’s direct-child scan and supported extensions—not a generic recursive file search.</p></div><span>${auditEffective} effective samples</span></div>${payload.datasets.map(report=>{
    if(report.error)return `<article class="dataset-audit-row error"><div><strong>Source ${report.index+1}</strong><small>${esc(report.kind||"source")} · unavailable</small></div><p>${esc(report.error)}</p></article>`;
    const captionBase=report.primary_count??report.media_count,coverage=captionBase?Math.round(report.caption_count/captionBase*100):0,attention=report.missing_caption_count+report.empty_caption_count+report.unreadable_caption_count+report.unreadable_count+report.ignored_nested_count,weight=auditEffective?Math.round(report.effective_samples/auditEffective*100):0;
    return `<article class="dataset-audit-row ${attention?"warning":"ok"}" data-audit-source="${report.index}" role="button" tabindex="0"><div class="audit-source-name"><strong>Source ${report.index+1} · ${esc(datasetBasename(state.dataset.datasets[report.index]?.source))}</strong><small>${report.trainer_usable_count} usable of ${report.media_count} · ${coverage}% captioned · ×${report.repeats} = ${report.effective_samples} · ${weight}% of epoch</small></div><div class="audit-badges"><span>${report.aspects?.landscape||0} landscape</span><span>${report.aspects?.portrait||0} portrait</span><span>${report.aspects?.square||0} square</span></div>${attention?`<p>${report.missing_caption_count?`${report.missing_caption_count} missing captions. `:""}${report.empty_caption_count?`${report.empty_caption_count} empty captions. `:""}${report.unreadable_count?`${report.unreadable_count} unreadable media. `:""}${report.ignored_nested_count?`${report.ignored_nested_count} nested files are ignored by Musubi. `:""}</p>`:`<p>All discovered media is readable and captioned.</p>`}</article>`;
  }).join("")}`;
  host.querySelectorAll("[data-audit-source]").forEach(row=>{const open=()=>{state.selectedDataset=Number(row.dataset.auditSource);state.datasetTab="media";state.datasetMediaPage=1;renderDataset(state.dataset,state.selectedDataset);loadDatasetMedia({skipFlush:true}).catch(error=>renderDatasetMediaError(error))};row.addEventListener("click",open);row.addEventListener("keydown",event=>{if(["Enter"," "].includes(event.key)){event.preventDefault();open()}})});
  setDatasetTab("health",{load:false});host.tabIndex=-1;host.focus({preventScroll:true});renderDatasetRail();renderDatasetOverview();toast("Full dataset audit complete.");
}
function openDatasetMedia(index){
  if(state.datasetCaptionDirty&&!confirm("Discard the unsaved caption changes?"))return;
  state.openDatasetMediaIndex=index;renderDatasetMediaInspector();
  if(!$("#dataset-media-dialog").open)$("#dataset-media-dialog").showModal();
}
function renderDatasetMediaInspector(){
  const item=state.datasetMedia?.items?.[state.openDatasetMediaIndex];if(!item)return;
  const url=`/api/dataset/media-file?token=${encodeURIComponent(item.token)}`;
  $("#dataset-inspector-media").innerHTML=!item.token?`<div class="media-placeholder large"><span>!</span><p>This file is missing or cannot be previewed.</p></div>`:item.preview_kind==="video"?`<video src="${url}" controls preload="metadata"></video>`:`<img src="${url}" alt="${esc(item.name)}">`;
  $("#dataset-inspector-kicker").textContent=`${item.kind.toUpperCase()} · ${item.role==="target"?"LAYER TARGET":"TRAINING ITEM"}`;
  $("#dataset-inspector-name").textContent=item.name;
  $("#dataset-inspector-meta").textContent=[item.width&&item.height?`${item.width}×${item.height}`:"",formatBytes(item.bytes),item.relative_path].filter(Boolean).join(" · ");
  const copy={eligible:["Ready for training","ok"],warning:["Training item needs review","warning"],excluded:["Excluded from image training","warning"],error:["Dataset loading will fail","error"],paired_target:["Paired layered target","ok"]}[item.training_state]||[item.training_state,"warning"];
  $("#dataset-inspector-status").innerHTML=`<div class="issue ${copy[1]}">${esc(copy[0])}${item.shared_caption?" · This sidecar is shared by another same-stem media file.":""}</div>`;
  const editor=$("#dataset-caption-editor");editor.value=item.caption||"";editor.disabled=!item.token||item.caption_state==="not_configured";
  $("#dataset-caption-help").textContent=item.caption_mode==="jsonl"?"Saving updates this JSONL record atomically; other lines and keys stay intact.":item.caption_state==="not_configured"?"Set a caption extension in Source settings before creating sidecars.":"Caption saves explicitly to the sidecar beside this media file.";
  $("#dataset-caption-state").textContent="No changes";$("#save-dataset-caption").disabled=true;state.datasetCaptionDirty=false;
  $("#dataset-inspector-controls").innerHTML=item.controls?.length?`<div class="inspector-controls"><strong>Paired controls</strong><div>${item.controls.map(control=>`<img src="/api/dataset/media-file?token=${encodeURIComponent(control.token)}" alt="${esc(control.name)}" title="${esc(control.name)}">`).join("")}</div></div>`:"";
  $("#dataset-media-position").textContent=`${state.openDatasetMediaIndex+1} of ${state.datasetMedia.items.length} on this page`;
  $("#dataset-media-prev").disabled=state.openDatasetMediaIndex<=0;$("#dataset-media-next").disabled=state.openDatasetMediaIndex>=state.datasetMedia.items.length-1;
}
function moveDatasetMediaInspector(delta){
  if(state.datasetCaptionDirty&&!confirm("Discard the unsaved caption changes?"))return;
  const next=state.openDatasetMediaIndex+delta;if(next<0||next>=state.datasetMedia.items.length)return;state.openDatasetMediaIndex=next;renderDatasetMediaInspector();
}
async function saveDatasetCaption(){
  const item=state.datasetMedia?.items?.[state.openDatasetMediaIndex];if(!item)return;
  const caption=$("#dataset-caption-editor").value,button=$("#save-dataset-caption");
  await withBusy(button,"Saving…",async()=>{
    await api("/api/dataset/caption",{method:"POST",body:JSON.stringify({token:item.token,caption,expected_revision:item.caption_revision})});
    state.datasetCaptionDirty=false;$("#dataset-caption-state").textContent="Saved";
    const name=item.name;await loadDatasetMedia({page:state.datasetMedia.page,skipFlush:true});
    const nextIndex=Math.max(0,state.datasetMedia.items.findIndex(candidate=>candidate.name===name));state.openDatasetMediaIndex=nextIndex;renderDatasetMediaInspector();
    toast(item.caption_mode==="jsonl"?"JSONL caption saved.":"Caption sidecar saved.");
  });
}

async function validateSettings() {
  if (!acceptRawSettings()) {
    go("settings");
    toast("Fix the raw settings JSON before validating or starting.", "error");
    return false;
  }
  const result = await api("/api/settings/validate",{method:"POST",body:JSON.stringify({settings:state.settings})});
  const entries = [...(result.errors||[]).map(x=>({...x,level:"error"})),...(result.warnings||[]).map(x=>({...x,level:"warning"}))];
  $("#settings-validation").innerHTML = entries.length ? entries.map(x=>`<div class="issue ${x.level}"><strong>${esc(x.key)}</strong> · ${esc(x.message)}</div>`).join("") : `<div class="issue ok">Ready to train. Preflight checks passed.</div>`;
  return !(result.errors || []).length;
}
function acceptRawSettings() {
  try {
    const parsed = JSON.parse($("#settings-json").value);
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("Settings must be an object.");
    state.settings = parsed; $("#json-error").innerHTML = ""; return true;
  } catch(e) { $("#json-error").innerHTML = `<div class="issue error">${esc(e.message)}</div>`; return false; }
}
async function saveSettings() {
  if (!acceptRawSettings()) {
    go("settings");
    toast("Fix the raw settings JSON before saving.", "error");
    return false;
  }
  await api("/api/settings",{method:"POST",body:JSON.stringify({settings:state.settings})});
  setDirty(false); toast("Training recipe saved.");
  return true;
}
async function previewCommands() {
  if (!acceptRawSettings()) {
    go("settings");
    return toast("Fix the raw settings JSON before previewing commands.", "error");
  }
  const plan = await api("/api/commands/preview",{method:"POST",body:JSON.stringify({settings:state.settings})});
  const lines = []; Object.entries(plan).forEach(([phase,commands]) => { if (!Array.isArray(commands)) return; lines.push(`${phase.toUpperCase()}\n`); commands.forEach(command=>lines.push(command.map(v=>/\s/.test(v)?JSON.stringify(v):v).join(" ")+"\n")); });
  $("#command-preview").textContent = lines.join("\n"); $("#command-dialog").showModal();
}
async function startJob() {
  if (state.settings.dataset_config && !await loadDatasetForSettings()) {
    go("datasets");
    return toast("The Dataset TOML could not be loaded. Choose a valid file before starting.", "error");
  }
  if (!await validateSettings()) return toast("Review the setup issues before starting.");
  if (!await saveSettings()) return;
  const payload = await api("/api/jobs/start",{method:"POST",body:JSON.stringify({settings:state.settings,run_cache:true})});
  lastLogId = 0; $("#live-log").textContent = ""; renderActive(payload.job); go("run"); toast("Musubi training started.");
}

function readLocalPreference(key,fallback){try{const value=localStorage.getItem(key);return value==null?fallback:value}catch(_){return fallback}}
function writeLocalPreference(key,value){try{localStorage.setItem(key,String(value))}catch(_){} }
let lastLogId = 0, latestProgressLine = "", followLog = readLocalPreference("musubi-log-follow","true")==="true";
function parseProgressLine(message){
  const text=String(message||"").trim(),match=text.match(/^steps:\s*(\d{1,3})%.*?\b(\d+)\s*\/\s*(\d+)\s*\[([^\]]+)\]/i);
  if(!match)return null;
  const timing=match[4].match(/^(\d+(?::\d+){1,2})<([^,]+),\s*([^,\]]+)/);
  return {text,percent:Number(match[1]),step:Number(match[2]),total:Number(match[3]),elapsed:timing?.[1]||"",eta:timing?.[2]?.trim()||"",rate:timing?.[3]?.trim()||""};
}
function updateLiveProgress(progress){
  if(!progress)return;latestProgressLine=progress.text;$("#live-progress").hidden=false;$("#live-progress-text").textContent=progress.text;
}
function keepLiveLogAtBottom(){
  const log=$("#live-log"),pane=log.closest("[data-run-pane=log]"),run=$("#run");
  if(!followLog||!run.classList.contains("active")||(!run.classList.contains("run-split-view")&&!pane?.classList.contains("active")))return;
  log.scrollTop=log.scrollHeight;
}
const LIVE_LOG_BOTTOM_TOLERANCE=32;
function isLiveLogAtBottom(log){return log.scrollHeight-log.scrollTop-log.clientHeight<=LIVE_LOG_BOTTOM_TOLERANCE}
function setFollowLog(enabled,{scroll=false,persist=true}={}){followLog=Boolean(enabled);setTerminalToggle($("#follow-log"),followLog);if(persist)writeLocalPreference("musubi-log-follow",followLog);if(followLog&&scroll){const log=$("#live-log");requestAnimationFrame(()=>{log.scrollTop=log.scrollHeight})}}
function appendLogEntries(entries){
  const log=$("#live-log"),durable=[];
  // Older runs and already-buffered Windows carriage-return artifacts may
  // have left blank rows in the terminal. Compact them when new output arrives
  // so the real tail remains reachable without changing meaningful log text.
  if(log.textContent.includes("\n\n"))log.textContent=log.textContent.split(/\r?\n/).filter(line=>line.trim()).join("\n");
  entries.forEach(entry=>{const message=String(entry.message??"").replaceAll("\r","").trimEnd();if(!message.trim())return;const progress=parseProgressLine(message);if(progress)updateLiveProgress(progress);else durable.push(message)});
  if(!durable.length)return;const wasNearBottom=isLiveLogAtBottom(log);if(log.textContent==="Waiting for a job…")log.textContent="";
  log.textContent+=durable.join("\n")+"\n";if(followLog&&(wasNearBottom||log.clientHeight===0))log.scrollTop=log.scrollHeight;requestAnimationFrame(()=>keepLiveLogAtBottom());
}
async function pollJob() {
  try {
    const payload = await api(`/api/jobs/active?after=${lastLogId}`); lastLogId = payload.last_log_id;
    if (payload.log.length) appendLogEntries(payload.log);
    renderActive(payload.active); renderMetrics(payload.active?.metrics || {},payload.active);
    if(state.promptPreview?.jobId&&payload.active?.id===state.promptPreview.jobId){
      const status=payload.active.status;
      if(["completed","failed","stopped"].includes(status)&&state.promptPreview.status!==status){
        state.promptPreview={...state.promptPreview,status,outputs:payload.active.sample_outputs||state.promptPreview.outputs||[],message:status==="completed"?"Preview ready":`Preview ${status}`};
        renderPlan();
        toast(status==="completed"?"Prompt preview finished. Open Samples to compare it.":`Prompt preview ${status}.`,status==="completed"?"info":"error");
      }
    }
  } catch(_) {}
}
function renderActive(job) {
  state.activeJob=job;
  if(job?.captured_thumbnails&&state.captureNoticeJob!==job.id){state.captureNoticeJob=job.id;toast(`${job.captured_thumbnails} tested prompt thumbnail${job.captured_thumbnails===1?"":"s"} added to the library.`)}
  const live = job && ["starting","running","stopping"].includes(job.status);
  $("#run-dot").classList.toggle("live", live); $("#stop-job").disabled = !live || job.status === "stopping";
  $("#active-title").textContent = job?.name || "Training control"; $("#active-subtitle").textContent = job ? `${job.phase} · command ${job.command_index}/${job.command_count}` : "No Musubi process is running.";
  $("#active-status").textContent = (job?.status || "idle").toUpperCase();
}
function durationToSeconds(value){const parts=String(value||"").split(":").map(Number);return parts.every(Number.isFinite)?parts.reduce((sum,part)=>sum*60+part,0):0}
function formatDuration(value){const seconds=Math.max(0,Math.round(Number(value)||0));if(!seconds)return "—";const h=Math.floor(seconds/3600),m=Math.floor(seconds%3600/60),s=seconds%60;return h?h+"h "+String(m).padStart(2,"0")+"m":m?m+"m "+String(s).padStart(2,"0")+"s":s+"s"}
function renderMetrics(m,job) {
  const progress=parseProgressLine(latestProgressLine),step=Number(m.step||progress?.step||0),total=Number(m.total_steps||progress?.total||0),pct=total?Math.min(100,step/total*100):0;
  const configuredEpochs=Number(job?.settings?.max_train_epochs||0),totalEpochs=Number(m.total_epochs||configuredEpochs||0);let epoch=Number(m.epoch||0);if(!epoch&&totalEpochs&&total)epoch=Math.min(totalEpochs,Math.floor(step/Math.ceil(total/totalEpochs))+1);
  $("#metric-percent").textContent=`${Math.round(pct)}%`; $("#progress-ring").style.background=`conic-gradient(var(--accent2) ${pct}%,var(--surface2) ${pct}%)`;
  $("#progress-ring").setAttribute("aria-valuenow", String(Math.round(pct)));
  $("#progress-ring").setAttribute("aria-valuetext", total ? `${step} of ${total} steps` : "Waiting to start");
  $("#metric-progress").textContent=total?`${step.toLocaleString()} of ${total.toLocaleString()} steps`:"Waiting to start";
  $("#metric-epoch").textContent=totalEpochs?"Epoch "+(epoch||1)+" of "+totalEpochs:"Configure a recipe, then launch it from Review.";
  if(total&&totalEpochs){const perEpoch=Math.ceil(total/totalEpochs),boundary=Math.min(total,Math.max(1,epoch)*perEpoch),remaining=Math.max(0,boundary-step);$("#metric-next-epoch").textContent=remaining?remaining.toLocaleString()+" steps · at "+boundary.toLocaleString():"Epoch boundary reached"}else $("#metric-next-epoch").textContent="Waiting for epoch data";
  const started=job?.started_at?Date.parse(job.started_at):NaN,wallElapsed=Number.isFinite(started)?Math.max(0,(Date.now()-started)/1000):0,barElapsed=durationToSeconds(progress?.elapsed),elapsed=barElapsed||wallElapsed;
  $("#metric-elapsed").textContent=formatDuration(elapsed);$("#metric-eta").textContent=progress?.eta||((step&&total&&elapsed)?formatDuration(elapsed/step*(total-step)):"—");$("#metric-rate").textContent=progress?.rate||((step&&elapsed)?(elapsed/step).toFixed(2)+"s / step":"—");
  $("#metric-loss").textContent=m.loss==null?"—":Number(m.loss).toFixed(5); $("#metric-depth").textContent=m.depth_loss==null?"—":Number(m.depth_loss).toFixed(5); $("#metric-dop").textContent=m.dop_loss==null?"—":Number(m.dop_loss).toFixed(5); drawLoss(m.loss_history||[]);
}
function drawLoss(history) {
  const canvas=$("#loss-canvas"),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height; ctx.clearRect(0,0,w,h); ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--line"); ctx.lineWidth=1;
  for(let i=1;i<5;i++){ctx.beginPath();ctx.moveTo(0,i*h/5);ctx.lineTo(w,i*h/5);ctx.stroke()}
  if(history.length<2){ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted");ctx.font="20px Segoe UI";ctx.fillText("Loss history will appear during training",35,h/2);return}
  const values=history.map(p=>Number(p[1])).filter(Number.isFinite),min=Math.min(...values),range=Math.max(...values)-min||1; ctx.beginPath();
  history.forEach((p,i)=>{const x=i/(history.length-1)*w,y=h-25-(Number(p[1])-min)/range*(h-50);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--accent2");ctx.lineWidth=3;ctx.stroke();
}
async function pollGpu() {
  try { const p=await api("/api/gpu"); if(!p.available||!p.devices.length)return $("#metric-vram").textContent="—"; const d=p.devices.reduce((a,b)=>b.memory_used>a.memory_used?b:a);$("#metric-vram").textContent=`${(d.memory_used/1073741824).toFixed(1)} / ${(d.memory_total/1073741824).toFixed(1)} GB`; } catch(_){}
}

async function loadSamples() {
  const p=await api(`/api/samples?output_dir=${encodeURIComponent(state.settings.output_dir||"")}&output_name=${encodeURIComponent(state.settings.output_name||"")}`); state.samples=p; $("#sample-count").textContent=p.groups.length;
  const host=$("#sample-groups");host.classList.toggle("empty",!p.groups.length);host.innerHTML=p.groups.length?p.groups.map((g,i)=>`<button class="sample-series" data-index="${i}"><strong>${esc(g.items[0]?.prefix||`Prompt ${i+1}`)} · Prompt ${esc(String(g.items[0]?.prompt_index??i).padStart(2,"0"))}</strong><small>${g.items.length} checkpoint${g.items.length===1?"":"s"}${g.items[0]?.seed!=null?` · seed ${esc(g.items[0].seed)}`:""}</small></button>`).join(""):"No sample series found.";
  host.querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>renderComparison(Number(b.dataset.index))));
  if(state.sampleMode==="gallery")renderSampleGallery();else if(p.groups.length)renderComparison(Math.min(compareState.group,p.groups.length-1));
}
function renderSampleGallery(){
  const items=[...state.samples.groups.flatMap(group=>group.items),...(state.samples.ungrouped||[])].sort((a,b)=>b.modified-a.modified);
  const media=item=>item.media_kind==="video"?`<video src="${item.url}" preload="metadata" muted playsinline></video>`:`<img src="${item.url}" loading="lazy" alt="">`;
  $("#compare-stage").innerHTML=`<div class="compare-head"><div><p class="kicker">ALL OUTPUTS</p><h2>Sample gallery</h2></div><span>${items.length} output${items.length===1?"":"s"}</span></div><div class="sample-gallery">${items.map((item,index)=>`<button class="gallery-card" data-index="${index}">${media(item)}<strong>${esc(item.sequence_label||item.name)}</strong><small>${esc(item.name)}</small></button>`).join("")}</div>`;
  $$(".gallery-card").forEach(button=>button.addEventListener("click",()=>openSamplePreview(items[Number(button.dataset.index)])));
}
function openSamplePreview(item){
  const image=$("#sample-preview-image"),video=$("#sample-preview-video"),isVideo=item.media_kind==="video";
  video.pause();video.removeAttribute("src");video.load();image.removeAttribute("src");
  image.hidden=isVideo;video.hidden=!isVideo;
  if(isVideo){video.src=item.url;video.load()}else image.src=item.url;
  $("#sample-preview-name").textContent=item.name;$("#sample-preview-dialog").showModal();
}
const compareState = {group:0,leftSequence:null,rightSequence:null,leftIndex:0,rightIndex:1,mode:"wipe",wipe:50,locked:false,videoProgress:0,videoPlaying:false,videoMuted:true,videoLoop:true};
function renderComparison(index) {
  compareState.group=index;
  $$(".sample-series").forEach((b,i)=>b.classList.toggle("active",i===index));
  const group=state.samples.groups[index],items=group.items;
  const restored=(sequence,fallback)=>{const found=sequence?items.findIndex(x=>x.sequence_kind===sequence[0]&&x.sequence===sequence[1]):-1;return found>=0?found:Math.max(0,Math.min(items.length-1,fallback))};
  compareState.leftIndex=restored(compareState.leftSequence,Math.max(0,items.length-2));
  compareState.rightIndex=restored(compareState.rightSequence,items.length-1);
  const isVideo=items[0]?.media_kind==="video";
  $("#compare-stage").innerHTML=`<div class="compare-head"><div><p class="kicker">${esc(items[0]?.prefix||"TRAINING SAMPLE")} · PROMPT ${esc(String(items[0]?.prompt_index??index).padStart(2,"0"))}</p><h2>Training progression</h2></div><div class="compare-tools"><button data-mode="wipe">Wipe slider</button><button data-mode="side">Side by side</button></div></div><div class="compare-nav"><button id="prev-prompt">← Previous prompt</button><span>${index+1} / ${state.samples.groups.length}</span><button id="next-prompt">Next prompt →</button><button id="prev-version">← Previous</button><button id="next-version">Next →</button><button id="wipe-lock">${compareState.locked?"🔒 Locked":"🔓 Follow pointer"}</button></div><div class="sample-meta"><span>${items.length} versions</span><span>${items[0]?.seed!=null?`Seed ${esc(items[0].seed)}`:"Seed not encoded"}</span><span>${isVideo?"Synchronized video comparison":"Keyboard and touch navigation enabled"}</span></div><div id="compare-viewport"></div>${isVideo?`<div class="video-compare-controls"><button id="video-play" type="button">▶ Play</button><input id="video-progress" type="range" min="0" max="1000" value="${Math.round(compareState.videoProgress*1000)}" aria-label="Video position"><span id="video-time">0:00 / 0:00</span><button id="video-mute" type="button">${compareState.videoMuted?"🔇 Muted":"🔊 Sound"}</button><button id="video-loop" type="button">${compareState.videoLoop?"↻ Loop on":"↻ Loop off"}</button><small id="video-compare-note" class="video-compare-note">Loading video details…</small></div>`:""}<div class="timeline"><select id="select-a" aria-label="Version A"></select><input id="sample-range" type="range" min="0" max="${items.length-1}" value="${compareState.rightIndex}" aria-label="Version B timeline"><select id="select-b" aria-label="Version B"></select></div>`;
  const options=items.map((item,i)=>`<option value="${i}">${esc(item.sequence_label||`${item.sequence_kind} ${item.sequence}`)} · ${esc(item.name)}</option>`).join("");
  $("#select-a").innerHTML=options;$("#select-b").innerHTML=options;$("#select-a").value=compareState.leftIndex;$("#select-b").value=compareState.rightIndex;
  const renderMode=()=>{
    const a=items[compareState.leftIndex],b=items[compareState.rightIndex],host=$("#compare-viewport");
    const previousMaster=host.querySelector('video[data-video-role="b"]');
    if(previousMaster&&Number.isFinite(previousMaster.duration)&&previousMaster.duration>0)compareState.videoProgress=previousMaster.currentTime/previousMaster.duration;
    compareState.leftSequence=[a.sequence_kind,a.sequence];compareState.rightSequence=[b.sequence_kind,b.sequence];
    $$("#compare-stage .compare-tools button").forEach(x=>x.classList.toggle("active",x.dataset.mode===compareState.mode));
    $("#wipe-lock").style.display=compareState.mode==="wipe"?"":"none";
    const media=(item,role)=>isVideo?`<video class="sync-video" data-video-role="${role}" src="${item.url}" preload="metadata" muted playsinline></video>`:`<img src="${item.url}" alt="Version ${role.toUpperCase()}">`;
    if(compareState.mode==="side"){
      host.innerHTML=`<div class="compare-main"><div class="compare-image">${media(a,"a")}<label>A · ${esc(a.sequence_label)}</label></div><div class="compare-image">${media(b,"b")}<label>B · ${esc(b.sequence_label)}</label></div></div>`;
    }else{
      host.innerHTML=`<div class="wipe-stage" id="wipe-stage" tabindex="0" role="slider" aria-label="Comparison reveal position" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(compareState.wipe)}"><div class="wipe-layer">${media(b,"b")}</div><div class="wipe-layer a" id="wipe-a">${media(a,"a")}</div><div class="wipe-divider" id="wipe-divider"></div><span class="wipe-label left">A · ${esc(a.sequence_label)}</span><span class="wipe-label right">B · ${esc(b.sequence_label)}</span></div>`;
      setWipe(compareState.wipe);bindWipe();
    }
    if(isVideo)bindVideoSync(host);
    $("#sample-range").value=compareState.rightIndex;
  };
  const setVersion=idx=>{compareState.rightIndex=Math.max(0,Math.min(items.length-1,idx));compareState.leftIndex=Math.max(0,compareState.rightIndex-1);$("#select-a").value=compareState.leftIndex;$("#select-b").value=compareState.rightIndex;renderMode()};
  const movePrompt=delta=>renderComparison((index+delta+state.samples.groups.length)%state.samples.groups.length);
  const setWipe=value=>{compareState.wipe=Math.max(0,Math.min(100,value));const layer=$("#wipe-a"),line=$("#wipe-divider"),stage=$("#wipe-stage");if(layer)layer.style.clipPath=`inset(0 ${100-compareState.wipe}% 0 0)`;if(line)line.style.left=`${compareState.wipe}%`;stage?.setAttribute("aria-valuenow",String(Math.round(compareState.wipe)))};
  function bindWipe(){const stage=$("#wipe-stage");if(!stage)return;let start=null;const move=e=>{if(compareState.locked)return;const point=e.touches?.[0]||e;const box=stage.getBoundingClientRect();setWipe((point.clientX-box.left)/box.width*100)};stage.addEventListener("pointermove",move);stage.addEventListener("pointerdown",move);stage.addEventListener("keydown",e=>{if(["ArrowLeft","ArrowRight","Home","End"].includes(e.key)){e.preventDefault();setWipe(e.key==="Home"?0:e.key==="End"?100:compareState.wipe+(e.key==="ArrowLeft"?-2:2))}});stage.addEventListener("dblclick",()=>{$("#wipe-lock").click()});stage.addEventListener("touchstart",e=>{start={x:e.touches[0].clientX,y:e.touches[0].clientY}},{passive:true});stage.addEventListener("touchend",e=>{if(!start)return;const dx=e.changedTouches[0].clientX-start.x,dy=e.changedTouches[0].clientY-start.y;if(Math.max(Math.abs(dx),Math.abs(dy))<40)return;if(Math.abs(dx)>Math.abs(dy))setVersion(compareState.rightIndex+(dx<0?1:-1));else movePrompt(dy<0?1:-1)},{passive:true})}
  function bindVideoSync(host){
    const master=host.querySelector('video[data-video-role="b"]'),slave=host.querySelector('video[data-video-role="a"]'),play=$("#video-play"),progress=$("#video-progress"),time=$("#video-time"),mute=$("#video-mute"),loop=$("#video-loop"),note=$("#video-compare-note");
    if(!master||!slave)return;
    const clock=value=>{if(!Number.isFinite(value))return "0:00";const seconds=Math.max(0,Math.floor(value));return `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,"0")}`};
    const setRatio=ratio=>{compareState.videoProgress=Math.max(0,Math.min(1,ratio||0));[master,slave].forEach(video=>{if(Number.isFinite(video.duration)&&video.duration>0)video.currentTime=compareState.videoProgress*video.duration});progress.value=String(Math.round(compareState.videoProgress*1000));time.textContent=`${clock(master.currentTime)} / ${clock(master.duration)}`};
    const align=force=>{if(!Number.isFinite(master.duration)||!Number.isFinite(slave.duration)||master.duration<=0||slave.duration<=0)return;const ratio=master.currentTime/master.duration,target=ratio*slave.duration;if(force||Math.abs(slave.currentTime-target)>.06)slave.currentTime=target;compareState.videoProgress=ratio;progress.value=String(Math.round(ratio*1000));time.textContent=`${clock(master.currentTime)} / ${clock(master.duration)}`};
    const updateDetails=()=>{if(master.readyState<1||slave.readyState<1)return;const sameSize=master.videoWidth===slave.videoWidth&&master.videoHeight===slave.videoHeight,durationDelta=Math.abs(master.duration-slave.duration);note.textContent=sameSize&&durationDelta<.05?`${master.videoWidth}×${master.videoHeight} · ${clock(master.duration)} · matched previews`:`Previews differ (${master.videoWidth}×${master.videoHeight}, ${clock(master.duration)} versus ${slave.videoWidth}×${slave.videoHeight}, ${clock(slave.duration)}); playback is aligned by progress.`;setRatio(compareState.videoProgress)};
    const applySound=()=>{master.muted=compareState.videoMuted;slave.muted=true;mute.textContent=compareState.videoMuted?"🔇 Muted":"🔊 B sound"};
    const applyLoop=()=>{master.loop=compareState.videoLoop;slave.loop=compareState.videoLoop;loop.textContent=compareState.videoLoop?"↻ Loop on":"↻ Loop off"};
    const pauseBoth=()=>{master.pause();slave.pause();compareState.videoPlaying=false;play.textContent="▶ Play"};
    const playBoth=async()=>{compareState.videoPlaying=true;play.textContent="❚❚ Pause";align(true);const results=await Promise.allSettled([slave.play(),master.play()]);if(results.some(result=>result.status==="rejected"))pauseBoth()};
    master.addEventListener("loadedmetadata",updateDetails);slave.addEventListener("loadedmetadata",updateDetails);master.addEventListener("timeupdate",()=>align(false));master.addEventListener("ended",()=>{if(!compareState.videoLoop)pauseBoth()});
    play.onclick=()=>compareState.videoPlaying?pauseBoth():playBoth();progress.oninput=()=>{const resume=compareState.videoPlaying;pauseBoth();setRatio(Number(progress.value)/1000);if(resume)playBoth()};mute.onclick=()=>{compareState.videoMuted=!compareState.videoMuted;applySound()};loop.onclick=()=>{compareState.videoLoop=!compareState.videoLoop;applyLoop()};
    applySound();applyLoop();setRatio(compareState.videoProgress);if(compareState.videoPlaying)playBoth();
  }
  $("#select-a").addEventListener("change",e=>{compareState.leftIndex=Number(e.target.value);renderMode()});$("#select-b").addEventListener("change",e=>{compareState.rightIndex=Number(e.target.value);renderMode()});$("#sample-range").addEventListener("input",e=>setVersion(Number(e.target.value)));
  $$("#compare-stage .compare-tools button").forEach(x=>x.addEventListener("click",()=>{compareState.mode=x.dataset.mode;renderMode()}));
  $("#prev-prompt").addEventListener("click",()=>movePrompt(-1));$("#next-prompt").addEventListener("click",()=>movePrompt(1));$("#prev-version").addEventListener("click",()=>setVersion(compareState.rightIndex-1));$("#next-version").addEventListener("click",()=>setVersion(compareState.rightIndex+1));
  $("#wipe-lock").addEventListener("click",()=>{compareState.locked=!compareState.locked;$("#wipe-lock").textContent=compareState.locked?"🔒 Locked":"🔓 Follow pointer";renderMode()});
  renderMode();
}
async function loadJobs() {
  const p=await api("/api/jobs");state.jobs=p.jobs||[];
  $("#recent-jobs").innerHTML=state.jobs.slice(0,4).map((j,index)=>`<button class="recent-job" data-recent-index="${index}"><span>${esc(j.name||j.output_name||"Unnamed")}</span><strong class="recent-status ${esc(j.status)}">${esc(j.status)}</strong></button>`).join("")||"Your recent Musubi jobs will appear here.";
  $$("#recent-jobs [data-recent-index]").forEach(button=>button.addEventListener("click",()=>{const job=state.jobs[Number(button.dataset.recentIndex)],row=document.createElement("div");row.dataset.source=job._source;row.dataset.index=job._history_index;showJobDetails(job,row)}));
  renderJobs();
}
function formatJobSpeed(value){return value==null?"—":`${Number(value).toFixed(3)} s/it`}
function jobRowMarkup(j,i){
  const key=`${j._source}:${j._history_index}`,selected=(state.jobComparison||[]).some(item=>item.key===key),perf=j.performance||{},speed=perf.median_seconds_per_iteration??perf.overall_seconds_per_iteration;
  return `<article class="job-row ${selected?"comparison-selected":""}" data-list-index="${i}" data-source="${esc(j._source)}" data-index="${j._history_index}"><button class="job-name"><strong>${esc(j.name||j.output_name||j.title||"Unnamed job")}</strong><small>${esc(j.mode||j.settings?.training_mode||j.settings_snapshot?.training_mode||"")} · ${esc(j.started_at?new Date(j.started_at).toLocaleString():"")}</small></button><span class="job-status ${esc(j.status)}">${esc(j.status)}</span><div><strong>${speed==null?esc((j.phase||j.kind||"").replaceAll("_"," ")):formatJobSpeed(speed)}</strong><small>${esc(j._source)} · ${esc(perf.quality||"no timing")}</small></div><div class="job-actions"><button class="quiet continue-job" title="Start additional training from this job in a new output">Continue as new</button>${j.kind==="training"&&["failed","stopped"].includes(j.status)?'<button class="quiet recover-job" title="Resume the verified saved optimizer and step position">Resume exact</button>':""}<button class="quiet more-job" aria-label="More actions for this job" aria-expanded="false">•••</button><div class="job-menu" role="menu" hidden><button role="menuitem" data-action="details">View performance & log</button><button role="menuitem" data-action="compare">${selected?"Remove from comparison":"Add to comparison"}</button><button role="menuitem" data-action="repeat">Repeat / edit as new</button><button role="menuitem" data-action="apply">Apply recipe</button>${j.settings_snapshot?.sample_prompts_data?.length||j.settings?.sample_prompts_data?.length?'<button role="menuitem" data-action="prompts">Import sample prompts</button>':""}${(j.mode||j.settings?.training_mode||j.settings_snapshot?.training_mode)==="Krea 2"?'<button role="menuitem" data-action="face">Refine face identity…</button>':""}<button role="menuitem" data-action="output">Open output</button><button role="menuitem" data-action="logs">Open TensorBoard / W&B folder</button><button role="menuitem" data-action="copy">Copy command</button></div></div></article>`;
}
function renderJobs(){
  const query=($("#job-search")?.value||"").trim().toLowerCase(),status=$("#job-status-filter")?.value||"";
  const filtered=(state.jobs||[]).filter(job=>{
    const snapshot=job.settings_snapshot||job.settings||{};
    const haystack=JSON.stringify([job.name,job.output_name,job.title,job.mode,snapshot.training_mode,job.kind,job.status]).toLowerCase();
    return (!query||haystack.includes(query))&&(!status||job.status===status);
  });
  const pageSize=innerWidth<=680?10:25,pageCount=Math.max(1,Math.ceil(filtered.length/pageSize));
  state.jobPage=Math.min(state.jobPage,pageCount-1);
  const jobs=filtered.slice(state.jobPage*pageSize,(state.jobPage+1)*pageSize);
  const host=$("#jobs-list");host.classList.toggle("empty",!filtered.length);$("#job-result-count").textContent=`${filtered.length} of ${(state.jobs||[]).length}`;
  host.innerHTML=filtered.length?`<div class="job-header" aria-hidden="true"><span>Run</span><span>Status</span><span>Performance</span><span>Actions</span></div>`+jobs.map((j,i)=>jobRowMarkup(j,i)).join("")+`<nav class="history-pager" aria-label="History pages"><button class="quiet" data-job-page="-1" ${state.jobPage===0?"disabled":""}>Previous</button><span>Page ${state.jobPage+1} of ${pageCount}</span><button class="quiet" data-job-page="1" ${state.jobPage>=pageCount-1?"disabled":""}>Next</button></nav>`:"No jobs match the current filters.";
  host.querySelectorAll(".job-row").forEach(row=>{const job=jobs[Number(row.dataset.listIndex)],menu=row.querySelector(".job-menu"),more=row.querySelector(".more-job");row.querySelector(".continue-job").addEventListener("click",()=>prepareJob(row,"continuation").catch(error=>toast(error.message,"error")));row.querySelector(".recover-job")?.addEventListener("click",()=>prepareJob(row,"recovery").catch(error=>toast(error.message,"error")));more.addEventListener("click",e=>{e.stopPropagation();$$(".job-menu").forEach(x=>{if(x!==menu)x.hidden=true});menu.hidden=!menu.hidden;more.setAttribute("aria-expanded",String(!menu.hidden));if(!menu.hidden)menu.querySelector("button")?.focus()});row.querySelector(".job-name").addEventListener("click",()=>showJobDetails(job,row));menu.querySelector('[data-action="details"]').addEventListener("click",()=>showJobDetails(job,row));menu.querySelector('[data-action="compare"]').addEventListener("click",()=>toggleJobComparison(job));menu.querySelector('[data-action="repeat"]').addEventListener("click",()=>repeatJob(job).catch(error=>toast(error.message,"error")));menu.querySelector('[data-action="apply"]').addEventListener("click",()=>applyJobSettings(job).catch(error=>toast(error.message,"error")));menu.querySelector('[data-action="prompts"]')?.addEventListener("click",()=>importJobPrompts(job));menu.querySelector('[data-action="face"]')?.addEventListener("click",()=>prepareFaceJob(row));menu.querySelector('[data-action="output"]').addEventListener("click",()=>openJobPath(row,"output"));menu.querySelector('[data-action="logs"]').addEventListener("click",()=>openJobPath(row,"logs"));menu.querySelector('[data-action="copy"]').addEventListener("click",()=>copyJobCommand(job))});
  host.querySelectorAll("[data-job-page]").forEach(button=>button.addEventListener("click",()=>{state.jobPage+=Number(button.dataset.jobPage);renderJobs();$("#jobs-list").scrollIntoView({block:"start"})}));
}
let detailJob=null,detailRow=null;
function drawJobSpeedChart(canvas,jobs){const ctx=canvas.getContext("2d"),box=canvas.getBoundingClientRect(),scale=devicePixelRatio||1;canvas.width=Math.max(300,Math.round(box.width*scale));canvas.height=Math.round(220*scale);ctx.scale(scale,scale);const w=canvas.width/scale,h=canvas.height/scale,pad=36,series=jobs.map(job=>job.performance?.speed_history||[]).filter(points=>points.length);ctx.clearRect(0,0,w,h);ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--line");ctx.beginPath();ctx.moveTo(pad,10);ctx.lineTo(pad,h-pad);ctx.lineTo(w-10,h-pad);ctx.stroke();if(!series.length){ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted");ctx.font="12px sans-serif";ctx.fillText("No measured per-step timing curve was saved for this historical job.",pad+15,h/2);return}const maxX=Math.max(...series.flat().map(p=>Number(p[0])||0),1),maxY=Math.max(...series.flat().map(p=>Number(p[1])||0),1),colors=["#51d6a2","#73a9ff"];series.forEach((points,index)=>{ctx.strokeStyle=colors[index%colors.length];ctx.lineWidth=1.6;ctx.beginPath();points.forEach((p,i)=>{const x=pad+(Number(p[0])/maxX)*(w-pad-15),y=10+(1-Number(p[1])/maxY)*(h-pad-15);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke()});ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted");ctx.font="11px sans-serif";ctx.fillText("step",w-38,h-10);ctx.save();ctx.translate(12,h/2);ctx.rotate(-Math.PI/2);ctx.fillText("s/it",0,0);ctx.restore()}
function showJobDetails(job,row){detailJob=job;detailRow=row;$("#job-dialog-title").textContent=job.name||job.output_name||job.title||"Unnamed job";const snapshot=job.settings_snapshot||job.settings||{},p=job.performance||{};const rows=[["Status",job.status],["Source",`${job._source||"desktop"} GUI`],["Mode",job.mode||snapshot.training_mode],["Started",job.started_at?new Date(job.started_at).toLocaleString():"—"],["Progress",job.progress||job.phase||`${p.step||0} / ${p.total_steps||0}`],["Measured median",formatJobSpeed(p.median_seconds_per_iteration)],["Recent median",formatJobSpeed(p.recent_seconds_per_iteration)],["Whole-job estimate",formatJobSpeed(p.overall_seconds_per_iteration)],["Output",snapshot.output_name||job.output_name||"—"],["Return code",job.return_code??"—"]];$("#job-dialog-summary").innerHTML=rows.map(([a,b])=>`<div class="review-row"><span>${esc(a)}</span><strong>${esc(b??"—")}</strong></div>`).join("");const note=$("#job-performance-note");note.className=`issue ${p.sample_count?"ok":"warning"}`;note.textContent=p.sample_count?`${p.sample_count} measured speed samples. Loading, cache preparation, previews, and saves are excluded from this curve.`:"This older job has no saved per-step timing curve. Its whole-job estimate includes loading, caching, previews, and checkpoint saves.";$("#job-console-log").hidden=true;$("#job-console-log").textContent="";$("#job-dialog-json").textContent=JSON.stringify(job,null,2);$("#job-dialog").showModal();requestAnimationFrame(()=>drawJobSpeedChart($("#job-speed-chart"),[job]))}
async function loadJobConsole(){if(!detailRow)return;const host=$("#job-console-log");try{const payload=await api(`/api/jobs/log?source=${encodeURIComponent(detailRow.dataset.source)}&index=${encodeURIComponent(detailRow.dataset.index)}`);host.textContent=payload.log;host.hidden=false;host.scrollTop=host.scrollHeight}catch(error){host.textContent=`No saved console log is available for this job.\n\n${error.message}`;host.hidden=false}}
function toggleJobComparison(job){state.jobComparison ||= [];const key=`${job._source}:${job._history_index}`,index=state.jobComparison.findIndex(item=>item.key===key);if(index>=0)state.jobComparison.splice(index,1);else{if(state.jobComparison.length>=2)state.jobComparison.shift();state.jobComparison.push({key,job})}const button=$("#compare-jobs");button.disabled=state.jobComparison.length!==2;button.textContent=`Compare selected (${state.jobComparison.length}/2)`;renderJobs()}
function showJobComparison(){const jobs=(state.jobComparison||[]).map(item=>item.job);if(jobs.length!==2)return;$("#job-compare-summary").innerHTML=jobs.map(job=>{const p=job.performance||{};return `<article><strong>${esc(job.name||job.output_name||job.title||"Unnamed")}</strong><small>${esc(job._source)} GUI · ${esc(job.mode||job.settings?.training_mode||job.settings_snapshot?.training_mode||"")}</small><p>Median: ${formatJobSpeed(p.median_seconds_per_iteration)}<br>Recent: ${formatJobSpeed(p.recent_seconds_per_iteration)}<br>Whole-job estimate: ${formatJobSpeed(p.overall_seconds_per_iteration)}<br>Samples: ${p.sample_count||0}</p></article>`}).join("");$("#job-compare-dialog").showModal();requestAnimationFrame(()=>drawJobSpeedChart($("#job-compare-chart"),jobs))}
async function replaySettings(job){const p=await api("/api/jobs/replay-settings",{method:"POST",body:JSON.stringify({source:job._source,index:Number(job._history_index)})});return structuredClone(p.settings||{})}
function splitRepeatName(name){let base=String(name||"").trim()||"run",generation=0,match;while((match=base.match(/-repeat(\d*)$/i))){generation+=Number(match[1]||1);base=base.slice(0,match.index).replace(/-+$/g,"")||"run"}return {base,generation}}
function nextRepeatName(name){const parsed=splitRepeatName(name);let generation=parsed.generation+1;for(const job of state.jobs||[]){const snapshot=job.settings_snapshot||job.settings||{},existing=splitRepeatName(snapshot.output_name||job.output_name||job.name);if(existing.base.toLowerCase()===parsed.base.toLowerCase())generation=Math.max(generation,existing.generation+1)}return `${parsed.base}-repeat${generation===1?"":generation}`}
async function repeatJob(job){const snapshot=await replaySettings(job);if(!Object.keys(snapshot).length)return toast("This older job has no complete settings snapshot.");if(!confirmWorkspaceReplacement("Load this job as a new editable run?"))return;discardWorkspaceDrafts();snapshot.resume_path="";snapshot.network_weights="";snapshot.starting_point_mode="new";snapshot.recovery_mode=false;snapshot.resume_exact_position=false;snapshot.output_name=nextRepeatName(snapshot.output_name||job.output_name||job.name||"run");state.settings=snapshot;renderGuided();renderAllSettings();sync();const datasetLoaded=await loadDatasetForSettings();go("setup");setStep("review");toast(datasetLoaded?"Repeat loaded as a new editable run.":"Repeat loaded, but its saved Dataset TOML could not be loaded. Choose a valid file before starting.",datasetLoaded?"info":"error")}
async function openJobPath(row,kind){try{const p=await api("/api/jobs/open-path",{method:"POST",body:JSON.stringify({source:row.dataset.source,index:Number(row.dataset.index),kind})});toast(`Opened ${p.opened}`)}catch(e){toast(e.message)}}
async function applyJobSettings(job){if(!confirmWorkspaceReplacement("Apply this job's saved recipe?"))return;const snapshot=await replaySettings(job);if(!Object.keys(snapshot).length)return toast("This older job has no complete settings snapshot.");discardWorkspaceDrafts();state.settings={...state.settings,...snapshot};renderGuided();renderAllSettings();sync();const datasetLoaded=await loadDatasetForSettings();go("setup");toast(datasetLoaded?"Job recipe applied for review.":"Recipe applied, but its saved Dataset TOML could not be loaded. Choose a valid file before starting.",datasetLoaded?"info":"error")}
function importJobPrompts(job){const prompts=(job.settings_snapshot||job.settings||{}).sample_prompts_data||[];state.settings.sample_prompts_data||=[];const known=new Set(state.settings.sample_prompts_data.map(x=>JSON.stringify(x)));let added=0;prompts.forEach(x=>{const key=JSON.stringify(x);if(!known.has(key)){state.settings.sample_prompts_data.push(structuredClone(x));known.add(key);added++}});sync();toast(`${added} sample prompt${added===1?"":"s"} imported.`)}
async function copyJobCommand(job){const value=Array.isArray(job.commands)?job.commands.map(x=>Array.isArray(x)?x.join(" "):x).join("\n"):job.command||"";if(!value)return toast("No command was recorded for this job.");await navigator.clipboard.writeText(value);toast("Command copied.")}
async function prepareJob(row,action){if(!confirmWorkspaceReplacement(action==="recovery"?"Load this exact recovery?":"Load this continuation?"))return;const p=await api(`/api/jobs/prepare-${action}`,{method:"POST",body:JSON.stringify({source:row.dataset.source,index:Number(row.dataset.index)})});discardWorkspaceDrafts();state.settings=p.settings;renderGuided();renderAllSettings();sync();const datasetLoaded=await loadDatasetForSettings();go("setup");setStep("review");const label=action==="recovery"?"Verified recovery loaded for review.":"Continuation loaded for review.";toast(datasetLoaded?label:`${label} Its saved Dataset TOML could not be loaded; choose a valid file before starting.`,datasetLoaded?"info":"error")}
async function prepareFaceJob(row){if(!confirmWorkspaceReplacement("Load this face-refinement continuation?"))return;try{const p=await api("/api/jobs/prepare-face",{method:"POST",body:JSON.stringify({source:row.dataset.source,index:Number(row.dataset.index)})});discardWorkspaceDrafts();state.settings=p.settings;renderGuided();renderAllSettings();sync();go("face");toast("Face-refinement continuation loaded for review.")}catch(e){toast(e.message)}}

async function initialize() {
  try {
    const payload=await api("/api/settings");state.settings=payload.settings;state.schema=payload.schema;
    if(!state.settings.face_refinement_config?.pose_plan){const defaults=await api("/api/face/defaults");state.settings.face_refinement_config=state.settings.face_refinement_config||{};state.settings.face_refinement_config.pose_plan=defaults.pose_plan;state.settings.face_refinement_config.face_model_dir ||= defaults.face_model_dir}
    const savedTheme=readLocalPreference("musubi-theme",state.settings.appearance_mode||"Dark");
    applyTheme(savedTheme,{syncSetting:true});
    renderGuided();renderAllSettings();sync(false);setDirty(false);$("#dataset-path").value=state.settings.dataset_config||"";
    $("#save-dataset").disabled=true;$("#inspect-dataset").disabled=true;
    $("#server-status").classList.add("online");$("#server-status").lastChild.textContent="Local service";
    loadJobs().catch(()=>{});
    const requested=location.hash.replace(/^#/,"");
    if(requested&&$(`.view#${CSS.escape(requested)}`))go(requested,{historyMode:"replace",focusHeading:false});
  } catch(e){$("#server-status").lastChild.textContent="Offline";toast(e.message)}
}

$$(".nav[data-view]").forEach(n=>{
  const label=n.querySelector("span:nth-of-type(2)")?.textContent?.trim()||n.dataset.view;
  n.title=label;n.setAttribute("aria-label",label);
  n.addEventListener("click",()=>go(n.dataset.view));
});
$$("[data-go]").forEach(n=>n.addEventListener("click",()=>go(n.dataset.go)));
$$(".recipe-step").forEach(n=>n.addEventListener("click",()=>setStep(n.dataset.step)));
$$(".model-tile").forEach(n=>n.addEventListener("click",()=>{selectMode(n.dataset.mode);go("setup");setStep("model")}));
function closeDialogSafely(dialog){
  if(dialog?.id==="dataset-media-dialog"&&state.datasetCaptionDirty&&!confirm("Discard the unsaved caption changes?"))return false;
  if(dialog?.id==="dataset-media-dialog")state.datasetCaptionDirty=false;
  if(dialog?.id==="sample-preview-dialog"){const video=$("#sample-preview-video"),image=$("#sample-preview-image");video?.pause();if(video)video.hidden=true;if(image)image.hidden=false}
  dialog?.close();return true;
}
$$("[data-close]").forEach(n=>n.addEventListener("click",()=>closeDialogSafely(document.getElementById(n.dataset.close))));
$$("dialog").forEach(dialog=>dialog.addEventListener("click",event=>{
  if(event.target!==dialog)return;
  const box=dialog.getBoundingClientRect();
  if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)closeDialogSafely(dialog);
}));
$("#dataset-media-dialog").addEventListener("cancel",event=>{if(state.datasetCaptionDirty&&!confirm("Discard the unsaved caption changes?"))event.preventDefault();else state.datasetCaptionDirty=false});
$("#plan-prompt-dialog").addEventListener("close",()=>{const index=state.openPromptIndex;renderPlan();setTimeout(()=>$(`.plan-prompt-card[data-prompt-index="${index}"] [data-action="edit"]`)?.focus({preventScroll:true}),0);state.planEditorReturnFocus=null});
$("#stage-editor-dialog").addEventListener("close",()=>{const index=state.openStageIndex;renderPlan();setTimeout(()=>$$(".plan-stage-card")[index]?.querySelector('[data-stage-action="edit"]')?.focus({preventScroll:true}),0);state.stageEditorReturnFocus=null});
$("#step-back").addEventListener("click",()=>moveStep(-1));$("#step-next").addEventListener("click",()=>moveStep(1));
$("#save-settings").addEventListener("click",()=>saveSettings().catch(e=>toast(e.message)));$("#save-all-settings").addEventListener("click",()=>saveSettings().catch(e=>toast(e.message)));
$("#load-settings-file").addEventListener("click",()=>$("#settings-file-input").click());$("#settings-file-input").addEventListener("change",async e=>{const file=e.target.files[0];if(!file)return;try{const parsed=JSON.parse(await file.text());if(!parsed||Array.isArray(parsed)||typeof parsed!=="object")throw new Error("Settings file must contain a JSON object.");if(!confirmWorkspaceReplacement(`Load ${file.name}?`))return;const schema=(await api("/api/settings")).schema;discardWorkspaceDrafts();state.settings=parsed;state.schema=schema;renderGuided();renderAllSettings();sync();toast(`Loaded ${file.name} for review.`)}catch(error){toast(error.message)}finally{e.target.value=""}});
$("#export-settings-file").addEventListener("click",()=>{const blob=new Blob([JSON.stringify(state.settings,null,2)],{type:"application/json"}),link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=`${state.settings.output_name||"musubi-settings"}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)});
$("#reset-settings").addEventListener("click",async()=>{if(!confirm("Restore every field to its default value? Saved files and training outputs are not deleted."))return;const payload=await api("/api/settings/defaults");state.settings=payload.settings;renderGuided();renderAllSettings();sync();toast("Defaults loaded for review.")});
$("#preview-button").addEventListener("click",()=>previewCommands().catch(e=>toast(e.message)));$("#validate-settings").addEventListener("click",()=>validateSettings().catch(e=>toast(e.message)));$("#start-from-review").addEventListener("click",()=>startJob().catch(e=>toast(e.message)));
$("#estimate-lora").addEventListener("click",async()=>{const mode=state.settings.training_mode,model=mode==="Krea 2"?state.settings.krea2_dit_model:mode==="MiniMax H3 (Experimental)"?state.settings.minimax_h3_dit_model:mode?.startsWith("Flux.2")?state.settings.flux2_dit_model:state.settings.dit_low_noise||state.settings.dit_high_noise;try{const result=await api("/api/estimate-lora",{method:"POST",body:JSON.stringify({model_path:model,mode,rank:state.settings.network_dim_low||state.settings.network_dim,network_type:state.settings.network_type,lokr_factor:state.settings.lokr_factor})});$("#lora-estimate").textContent=`Estimated adapter size: ${result.formatted}` }catch(e){toast(e.message)}});
$("#apply-workspace").addEventListener("click",async()=>{try{const result=await api("/api/workspace/apply",{method:"POST",body:JSON.stringify({root:state.settings.project_root})});state.settings.output_dir=result.output_dir;state.settings.logging_dir=result.logging_dir;state.settings.convert_output_dir=result.output_dir;renderGuided();renderAllSettings();sync();toast("Workspace models and log folders are ready.")}catch(e){toast(e.message)}});
$("#setting-search").addEventListener("input",filterSettings);$("#show-all-modes").addEventListener("change",filterSettings);$("#settings-json").addEventListener("change",()=>{if(acceptRawSettings()){renderGuided();renderAllSettings();sync()}});
$("#load-dataset").addEventListener("click",()=>loadDatasetDocument());
$("#dataset-path").addEventListener("keydown",event=>{if(event.key==="Enter"){event.preventDefault();loadDatasetDocument()}});
$("#dataset-source").addEventListener("input",()=>{state.datasetFormDirty=false;state.datasetRawDirty=true;state.samplingEstimate=null;renderSamplingEstimate();setDatasetDirty(true)});
$("#browse-dataset").addEventListener("click",()=>withBusy($("#browse-dataset"),"Choosing…",async()=>{const result=await api("/api/path/select",{method:"POST",body:JSON.stringify({kind:"file",initial:$("#dataset-path").value})});if(result.path){$("#dataset-path").value=result.path;await loadDatasetDocument()}}).catch(e=>toast(e.message,"error")));
$("#inspect-dataset").addEventListener("click",()=>withBusy($("#inspect-dataset"),"Auditing…",inspectDataset).catch(e=>toast(e.message,"error")));
$("#validate-dataset").addEventListener("click",()=>parseDatasetRawDraft({announce:true}).catch(error=>toast(error.message,"error")));
$("#dataset-defaults").addEventListener("click",selectDatasetDefaults);
$("#estimate-epoch-steps")?.addEventListener("click",estimateSamplingSteps);
$$("[data-dataset-tab]").forEach((button,index,buttons)=>{
  button.addEventListener("click",async()=>{if(button.disabled)return;try{await flushDatasetDraft();setDatasetTab(button.dataset.datasetTab)}catch(error){toast(error.message,"error")}});
  button.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;event.preventDefault();const enabled=buttons.filter(item=>!item.disabled),current=enabled.indexOf(button),target=event.key==="Home"?0:event.key==="End"?enabled.length-1:(current+(event.key==="ArrowLeft"?-1:1)+enabled.length)%enabled.length;enabled[target]?.click();enabled[target]?.focus()});
});
$$("[data-run-dataset-audit]").forEach(button=>button.addEventListener("click",()=>withBusy(button,"Auditing…",inspectDataset).catch(error=>toast(error.message,"error"))));
$("#dataset-media-search").addEventListener("input",event=>{state.datasetMediaQuery=event.target.value;clearTimeout(datasetMediaTimer);datasetMediaTimer=setTimeout(()=>loadDatasetMedia({page:1}).catch(error=>renderDatasetMediaError(error)),350)});
$("#dataset-media-filter").addEventListener("change",event=>{state.datasetMediaFilter=event.target.value;loadDatasetMedia({page:1}).catch(error=>renderDatasetMediaError(error))});
$("#open-dataset-source").addEventListener("click",event=>withBusy(event.currentTarget,"Opening…",async()=>{await flushDatasetDraft();const payload=await api("/api/dataset/open-source",{method:"POST",body:JSON.stringify({text:$("#dataset-source").value,index:state.selectedDataset})});toast(`Opened ${payload.opened}`)}).catch(error=>toast(error.message,"error")));
$("#refresh-dataset-media").addEventListener("click",event=>withBusy(event.currentTarget,"Refreshing…",()=>loadDatasetMedia({page:state.datasetMediaPage})).catch(error=>renderDatasetMediaError(error)));
$("#dataset-caption-editor").addEventListener("input",()=>{state.datasetCaptionDirty=true;$("#dataset-caption-state").textContent="Unsaved caption";$("#save-dataset-caption").disabled=false});
$("#dataset-media-prev").addEventListener("click",()=>moveDatasetMediaInspector(-1));$("#dataset-media-next").addEventListener("click",()=>moveDatasetMediaInspector(1));
$("#save-dataset-caption").addEventListener("click",()=>saveDatasetCaption().catch(error=>toast(error.message,"error")));
$("#add-image-dataset").addEventListener("click",()=>{state.datasetTab="settings";mutateDataset("/api/dataset/add",{kind:"image"},state.dataset?.datasets?.length||0)});
$("#add-video-dataset").addEventListener("click",()=>{state.datasetTab="settings";mutateDataset("/api/dataset/add",{kind:"video"},state.dataset?.datasets?.length||0)});
$("#save-dataset").addEventListener("click",()=>withBusy($("#save-dataset"),"Saving…",saveDatasetDocument).catch(error=>{if(error.status===409){$("#dataset-document-state").textContent="Disk changed";$("#dataset-document-state").classList.add("dirty")}toast(error.message,"error")}));
$("#refresh-samples").addEventListener("click",()=>loadSamples().catch(e=>toast(e.message)));$("#refresh-jobs").addEventListener("click",()=>loadJobs().catch(e=>toast(e.message)));
$("#job-search").addEventListener("input",()=>{state.jobPage=0;renderJobs()});$("#job-status-filter").addEventListener("change",()=>{state.jobPage=0;renderJobs()});
$("#import-jobs").addEventListener("click",()=>api("/api/jobs/import-found",{method:"POST",body:"{}"}).then(p=>{toast(`${p.added} job folder${p.added===1?"":"s"} imported.`);loadJobs()}).catch(e=>toast(e.message)));
$("#clear-jobs").addEventListener("click",()=>{if(!confirm("Delete all locally recorded web and desktop job-history entries? Training outputs are not deleted."))return;api("/api/jobs/clear",{method:"POST",body:"{}"}).then(()=>{toast("Local job history cleared.");loadJobs()}).catch(e=>toast(e.message))});
$$("[data-sample-mode]").forEach(button=>button.addEventListener("click",()=>{state.sampleMode=button.dataset.sampleMode;$$("[data-sample-mode]").forEach(x=>x.classList.toggle("active",x===button));if(state.samples){if(state.sampleMode==="gallery")renderSampleGallery();else if(state.samples.groups.length)renderComparison(Math.min(compareState.group,state.samples.groups.length-1))}}));
$("#stop-job").addEventListener("click",async()=>{try{renderActive((await api("/api/jobs/stop",{method:"POST",body:"{}"})).job);toast("Stop requested.")}catch(e){toast(e.message)}});$("#clear-log").addEventListener("click",()=>{$("#live-log").textContent="";latestProgressLine="";$("#live-progress").hidden=true});
$("#copy-log").addEventListener("click",()=>navigator.clipboard.writeText($("#live-log").textContent+(latestProgressLine?"\n"+latestProgressLine:"")).then(()=>toast("Console output copied.")));
function setTerminalToggle(button,active){button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active))}
const wrapLog=readLocalPreference("musubi-log-wrap","true")==="true";$("#live-log").classList.toggle("wrap-lines",wrapLog);setTerminalToggle($("#wrap-log"),wrapLog);setTerminalToggle($("#follow-log"),followLog);
$("#wrap-log").addEventListener("click",event=>{const active=!$("#live-log").classList.contains("wrap-lines");$("#live-log").classList.toggle("wrap-lines",active);setTerminalToggle(event.currentTarget,active);writeLocalPreference("musubi-log-wrap",active)});
$("#follow-log").addEventListener("click",()=>setFollowLog(!followLog,{scroll:!followLog}));
$("#live-log").addEventListener("scroll",event=>{const log=event.currentTarget;if(isLiveLogAtBottom(log)){if(!followLog)setFollowLog(true)}else if(followLog)setFollowLog(false)});
function setRunSplitRatio(ratio,persist=true){const value=Math.min(.75,Math.max(.25,Number(ratio)||.5)),percent=Math.round(value*100),stack=$("#run-panel-stack"),divider=$("#run-split-divider");stack.style.gridTemplateColumns=`calc(${percent}% - 5px) 10px calc(${100-percent}% - 5px)`;divider.setAttribute("aria-valuenow",String(percent));if(persist)writeLocalPreference("musubi-run-split-ratio",value)}
function setRunSplitView(enabled,persist=true){const run=$("#run"),button=$("#toggle-run-split");run.classList.toggle("run-split-view",enabled);button.setAttribute("aria-pressed",String(enabled));button.textContent=enabled?"Use tabs":"Show split view";if(persist)writeLocalPreference("musubi-run-split",enabled)}
const initialRunSplit=readLocalPreference("musubi-run-split","false")==="true";setRunSplitView(initialRunSplit,false);setRunSplitRatio(Number(readLocalPreference("musubi-run-split-ratio","0.5")),false);
$("#toggle-run-split").addEventListener("click",()=>setRunSplitView(!$("#run").classList.contains("run-split-view")));
const runDivider=$("#run-split-divider");runDivider.addEventListener("pointerdown",event=>{if(!$("#run").classList.contains("run-split-view"))return;event.preventDefault();const stack=$("#run-panel-stack"),rect=stack.getBoundingClientRect();const move=moveEvent=>setRunSplitRatio((moveEvent.clientX-rect.left)/rect.width,false),stop=()=>{window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",stop);const value=Number(runDivider.getAttribute("aria-valuenow"))/100;writeLocalPreference("musubi-run-split-ratio",value)};window.addEventListener("pointermove",move);window.addEventListener("pointerup",stop)});runDivider.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();const current=Number(runDivider.getAttribute("aria-valuenow"))/100;setRunSplitRatio(current+(event.key==="ArrowRight"?.05:-.05))});
function bindTabs(buttonSelector,paneSelector,buttonKey,paneKey){
  const buttons=$$(buttonSelector),panes=$$(paneSelector);
  buttons[0]?.parentElement?.setAttribute("role","tablist");
  buttons.forEach((button,index)=>{
    const pane=panes.find(item=>item.dataset[paneKey]===button.dataset[buttonKey]);
    button.setAttribute("role","tab");button.id||=`tab-${buttonKey}-${index}`;
    if(pane){pane.setAttribute("role","tabpanel");pane.setAttribute("aria-labelledby",button.id)}
    const activate=()=>{buttons.forEach(x=>{const active=x===button;x.classList.toggle("active",active);x.setAttribute("aria-selected",String(active));x.tabIndex=active?0:-1});panes.forEach(x=>x.classList.toggle("active",x.dataset[paneKey]===button.dataset[buttonKey]))};
    button.addEventListener("click",activate);
    button.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;event.preventDefault();const target=event.key==="Home"?0:event.key==="End"?buttons.length-1:(index+(event.key==="ArrowLeft"?-1:1)+buttons.length)%buttons.length;buttons[target].click();buttons[target].focus()});
  });
  buttons.find(button=>button.classList.contains("active"))?.click();
}
bindTabs("[data-run-tab]","[data-run-pane]","runTab","runPane");
$$('[data-run-tab]').forEach(button=>button.addEventListener("click",()=>{if($("#run").classList.contains("run-split-view"))setRunSplitView(false)}));
$("[data-run-tab=log]").addEventListener("click",()=>requestAnimationFrame(()=>keepLiveLogAtBottom()));
bindTabs("[data-plan-tab]","[data-plan-pane]","planTab","planPane");
bindTabs("[data-face-step]","[data-face-pane]","faceStep","facePane");
$("#use-stages").addEventListener("change",e=>{state.settings.use_staged_training=e.target.checked;renderPlan();sync()});
$("#add-prompt").addEventListener("click",addPlanPrompt);
$("#open-prompt-library").addEventListener("click",()=>openPromptLibrary().catch(e=>toast(e.message)));$("#library-search").addEventListener("input",renderLibrary);$("#save-prompts-library").addEventListener("click",()=>api("/api/prompt-library/import",{method:"POST",body:JSON.stringify({prompts:state.settings.sample_prompts_data})}).then(p=>toast(`${p.added} added, ${p.merged} already present.`)).catch(e=>toast(e.message)));
$("#library-save").addEventListener("click",async()=>{const entry=libraryEntries.find(item=>item.id===$("#library-edit-id").value);if(!entry)return;const promptData={...entry.prompt_data,prompt:$("#library-edit-prompt").value};try{await api("/api/prompt-library/update",{method:"POST",body:JSON.stringify({id:entry.id,name:$("#library-edit-name").value,prompt_data:promptData,collection:$("#library-edit-collection").value,tags:$("#library-edit-tags").value.split(",").map(x=>x.trim()).filter(Boolean)})});$("#prompt-editor-dialog").close();await openPromptLibrary();toast("Library prompt updated.")}catch(e){toast(e.message)}});
$("#library-delete").addEventListener("click",async()=>{const id=$("#library-edit-id").value;if(!confirm("Delete this library prompt and its stored test thumbnails?"))return;try{await api("/api/prompt-library/delete",{method:"POST",body:JSON.stringify({id})});$("#prompt-editor-dialog").close();await openPromptLibrary();toast("Library prompt deleted.")}catch(e){toast(e.message)}});
$("#preview-prompts").addEventListener("click",()=>{const prompts=state.settings.sample_prompts_data||[],indices=prompts.map((prompt,index)=>prompt.enabled!==false?index:-1).filter(index=>index>=0),enabled=indices.map(index=>prompts[index]);if(!enabled.length)return toast("Include at least one prompt to preview.");startPromptPreview(enabled,{indices,stayInPlan:true})});
$("#enable-prompts").addEventListener("click",()=>{state.settings.sample_prompts_data.forEach(x=>x.enabled=true);renderPlan();sync()});$("#disable-prompts").addEventListener("click",()=>{state.settings.sample_prompts_data.forEach(x=>x.enabled=false);renderPlan();sync()});
$("#add-stage").addEventListener("click",addPlanStage);
$("#save-plan").addEventListener("click",()=>saveTrainingPlan().catch(e=>toast(e.message,"error")));$("#copy-summary").addEventListener("click",()=>navigator.clipboard.writeText($("#training-summary").textContent).then(()=>toast("Settings summary copied.")));
$("#plan-prompt-preview").addEventListener("click",()=>{const prompt=state.settings.sample_prompts_data?.[state.openPromptIndex];if(prompt)startPromptPreview([{...prompt,enabled:true}],{indices:[state.openPromptIndex],stayInPlan:true})});
$("#save-face").addEventListener("click",()=>saveSettings().catch(e=>toast(e.message)));$("#add-face-prompt").addEventListener("click",()=>{faceConfig().prompts.push("portrait photo of {trigger}");renderFaceWorkspace();sync()});
$("#add-face-stage").addEventListener("click",addOrUpdateFaceStage);$("#review-face-stage").addEventListener("click",reviewFaceStage);
$("#download-face-models").addEventListener("click",async()=>{
  const config=faceConfig();
  if(!config.license_acknowledged)return toast("Acknowledge the third-party model license before downloading.");
  if(!confirm("Download the AntelopeV2 detection and recognition models (about 280 MB) into the selected model folder?"))return;
  const button=$("#download-face-models");button.disabled=true;button.textContent="Downloading AntelopeV2…";
  try{const payload=await api("/api/face/models/download",{method:"POST",body:JSON.stringify({face_model_dir:config.face_model_dir})});config.face_model_dir=payload.model_dir;renderFaceWorkspace();sync();toast("AntelopeV2 models are ready.")}
  catch(e){toast(e.message)}finally{button.disabled=false;button.textContent="Download models to selected folder"}
});
$("#apply-pose-preset").addEventListener("click",async()=>{try{const config=faceConfig(),payload=await api("/api/face/pose-preset",{method:"POST",body:JSON.stringify({preset:$("#pose-preset").value,plan:config.pose_plan})});config.pose_plan=payload.pose_plan;config.pose_aware=true;renderFaceWorkspace();sync();toast("Pose preset applied.")}catch(e){toast(e.message)}});
$("#import-pose-prompts").addEventListener("click",()=>$("#pose-prompt-file").click());
$("#pose-prompt-file").addEventListener("change",async e=>{const file=e.target.files[0];if(!file)return;try{const raw=await file.text(),parsed=file.name.toLowerCase().endsWith(".json")?JSON.parse(raw):raw.split(/\r?\n/),lines=Array.isArray(parsed)?parsed:(parsed.prompts||[]),buckets=faceConfig().pose_plan.buckets;let added=0;lines.forEach(value=>{const line=String(typeof value==="object"?value.prompt||"":value).trim(),match=line.match(/^\[([a-z_]+)\]/i);if(!line||!match||!buckets[match[1]])return;const prompts=buckets[match[1]].prompts||=[];if(!prompts.includes(line)){prompts.push(line);added++}});renderFaceWorkspace();sync();toast(`${added} pose-tagged prompt${added===1?"":"s"} imported.`)}catch(error){toast(error.message)}finally{e.target.value=""}});
$("#run-face-preflight").addEventListener("click",async()=>{const button=$("#run-face-preflight"),config=faceConfig();button.disabled=true;button.textContent="Analyzing…";try{config.preflight_report=await api("/api/face/preflight",{method:"POST",body:JSON.stringify({reference_dir:config.reference_dir,face_model_dir:config.face_model_dir})});sync();await saveSettings();renderFaceWorkspace();toast("Reference analysis completed.")}catch(e){toast(e.message)}finally{button.disabled=false;button.textContent="Analyze references"}});
async function startFaceEvaluation(comparison){
  const config=faceConfig(),baseline=comparison?config.evaluation_baseline_result||"":"";try{const payload=await api("/api/face/evaluate",{method:"POST",body:JSON.stringify({settings:state.settings,input_lora:config.input_lora,baseline_result:baseline})});if(!comparison){config.evaluation_baseline_result=payload.result}else{config.evaluation_last_result=payload.result}sync();await saveSettings();lastLogId=0;renderActive(payload.job);go("run");toast(`${payload.cases} fixed evaluation cases started.`)}catch(e){toast(e.message)}
}
async function loadFaceResult(){const config=faceConfig(),path=config.evaluation_last_result||config.evaluation_baseline_result;if(!path)return toast("No saved evaluation result path is available.");try{const payload=await api("/api/face/result",{method:"POST",body:JSON.stringify({path})}),poses=payload.poses||{},cases=payload.cases||[],deltas=payload.deltas||{};loadedFaceResult=payload;$("#build-weak-pose-plan").disabled=false;$("#face-eval-result").classList.remove("empty");$("#face-eval-result").innerHTML=`<div class="review-summary">${Object.entries(poses).map(([pose,m])=>{const delta=deltas[pose]?.overall_similarity;return `<div class="review-row"><span>${esc(pose.replaceAll("_"," "))} · ${m.samples} samples</span><strong>identity ${m.overall_similarity==null?"—":Number(m.overall_similarity).toFixed(3)}${delta==null?"":` · Δ ${delta>=0?"+":""}${Number(delta).toFixed(3)}`} · pose ${m.pose_success_rate==null?"—":Math.round(m.pose_success_rate*100)+"%"}</strong></div>`}).join("")}</div><div class="sample-gallery">${cases.filter(item=>item.image).map((item,index)=>`<button class="gallery-card" data-face-case="${index}"><img src="/api/evaluation-image?path=${encodeURIComponent(item.image)}" loading="lazy" alt=""><strong>${esc(item.pose)} → ${esc(item.actual_pose||"not detected")}</strong><small>identity ${item.overall_similarity==null?"—":Number(item.overall_similarity).toFixed(3)} · seed ${esc(item.seed)}</small></button>`).join("")}</div><details class="raw-settings"><summary>Complete evaluation data</summary><pre>${esc(JSON.stringify(payload,null,2))}</pre></details>`;$$("[data-face-case]").forEach(button=>button.addEventListener("click",()=>{const item=cases.filter(x=>x.image)[Number(button.dataset.faceCase)];$("#sample-preview-image").src=`/api/evaluation-image?path=${encodeURIComponent(item.image)}`;$("#sample-preview-name").textContent=`${item.pose} · seed ${item.seed}`;$("#sample-preview-dialog").showModal()}))}catch(e){toast(e.message)}}
$("#build-weak-pose-plan").addEventListener("click",async()=>{if(!loadedFaceResult)return;try{const config=faceConfig(),payload=await api("/api/face/weak-plan",{method:"POST",body:JSON.stringify({result:loadedFaceResult,plan:config.pose_plan,target_similarity:config.stop_similarity??.55,reference_counts:config.preflight_report?.pose_bucket_counts||{},min_references:config.pose_min_references??2})});config.pose_plan=payload.pose_plan;config.pose_aware=true;renderFaceWorkspace();sync();$$("[data-face-step]").find(x=>x.dataset.faceStep==="recipe")?.click();toast(payload.warnings?.length?payload.warnings.join(" "):"Weak-pose training plan created.")}catch(e){toast(e.message)}});
$("#face-baseline").addEventListener("click",()=>startFaceEvaluation(false));$("#face-compare").addEventListener("click",()=>startFaceEvaluation(true));
$("#load-face-result").addEventListener("click",loadFaceResult);
$("#open-face-results").addEventListener("click",async()=>{const config=faceConfig(),path=config.evaluation_last_result||config.evaluation_baseline_result;if(!path)return toast("No saved evaluation result is available.");try{const payload=await api("/api/face/open-results",{method:"POST",body:JSON.stringify({path})});toast(`Opened ${payload.opened}`)}catch(e){toast(e.message)}});
$("#start-conversion").addEventListener("click",async()=>{try{const payload=await api("/api/tools/convert",{method:"POST",body:JSON.stringify({input:state.settings.convert_lora_path,output_dir:state.settings.convert_output_dir,target:state.settings.convert_target||"default"})});lastLogId=0;renderActive(payload.job);go("run");toast(`Converting to ${payload.output}`)}catch(e){toast(e.message)}});
$("#run-accelerate-config").addEventListener("click",()=>api("/api/tools/accelerate-config",{method:"POST",body:"{}"}).then(()=>toast("Accelerate terminal opened.")).catch(e=>toast(e.message)));
$("#theme-toggle").addEventListener("click",()=>applyTheme(document.documentElement.dataset.theme==="light"?"Dark":"Light",{markChanged:true}));
$("#open-desktop").addEventListener("click",()=>api("/api/legacy/start",{method:"POST",body:"{}"}).then(()=>toast("Classic GUI opened.")).catch(e=>toast(e.message)));
$("#return-review").addEventListener("click",()=>{go("setup");setStep("review")});
$("#save-state").addEventListener("click",async()=>{
  const button=$("#save-state");if(button.dataset.busy==="true")return;button.dataset.busy="true";button.disabled=true;button.setAttribute("aria-busy","true");
  try{
    const hadDatasetDraft=state.datasetDirty||state.datasetFormDirty;
    if(hadDatasetDraft)await saveDatasetDocument({announce:false});
    if(state.dirty)await saveSettings();else if(hadDatasetDraft)toast("Dataset TOML saved.");
  }catch(error){toast(error.message,"error")}
  finally{button.dataset.busy="false";button.disabled=false;button.removeAttribute("aria-busy");updateSaveState()}
});
document.addEventListener("click",event=>{
  if(!event.target.closest(".job-actions"))$$(".job-menu").forEach(menu=>{menu.hidden=true;menu.closest(".job-actions")?.querySelector(".more-job")?.setAttribute("aria-expanded","false")});
  const actionMenu=event.target.closest(".action-menu"),itemMenu=event.target.closest(".item-menu");
  $$(".action-menu[open]").forEach(menu=>{if(menu!==actionMenu)menu.open=false});
  $$(".item-menu[open]").forEach(menu=>{if(menu!==itemMenu)menu.open=false});
});
document.addEventListener("keydown",event=>{
  const mediaDialogOpen=$("#dataset-media-dialog").open;
  if(event.ctrlKey&&event.key.toLowerCase()==="s"){event.preventDefault();if(mediaDialogOpen){if(state.datasetCaptionDirty)$("#save-dataset-caption").click()}else if(state.activeView==="datasets")$("#save-dataset").click();else saveSettings().catch(e=>toast(e.message));return}
  if(event.ctrlKey&&event.key.toLowerCase()==="o"){event.preventDefault();$("#settings-file-input").click();return}
  if(event.ctrlKey&&event.key==="Enter"&&state.activeView==="setup"&&state.step==="review"&&!document.querySelector("dialog[open]")){event.preventDefault();startJob().catch(e=>toast(e.message));return}
  if(event.key==="Escape"){$$(".job-menu").forEach(menu=>{menu.hidden=true;menu.closest(".job-actions")?.querySelector(".more-job")?.setAttribute("aria-expanded","false")});$$(".action-menu[open],.item-menu[open]").forEach(menu=>menu.open=false)}
  if(mediaDialogOpen&&!event.target.matches("input,select,textarea")){if(event.key==="ArrowLeft"){event.preventDefault();moveDatasetMediaInspector(-1)}if(event.key==="ArrowRight"){event.preventDefault();moveDatasetMediaInspector(1)}return}
  if(!$("#samples").classList.contains("active")||!state.samples?.groups?.length||event.target.matches("input,select,textarea"))return;
  if(event.key==="ArrowLeft")$("#prev-version")?.click();
  if(event.key==="ArrowRight")$("#next-version")?.click();
  if(event.key==="ArrowUp")$("#prev-prompt")?.click();
  if(event.key==="ArrowDown")$("#next-prompt")?.click();
});
$("#job-view-console").addEventListener("click",()=>loadJobConsole());$("#job-open-output").addEventListener("click",()=>detailRow&&openJobPath(detailRow,"output"));$("#job-open-logs").addEventListener("click",()=>detailRow&&openJobPath(detailRow,"logs"));$("#job-dialog-copy").addEventListener("click",()=>detailJob&&copyJobCommand(detailJob));$("#compare-jobs").addEventListener("click",showJobComparison);
applyTheme(localStorage.getItem("musubi-theme")||"Dark",{syncSetting:false,persist:false});
window.addEventListener("beforeunload",event=>{if(!state.dirty&&!state.datasetDirty&&!state.datasetFormDirty&&!state.datasetRawDirty&&!state.datasetCaptionDirty)return;event.preventDefault();event.returnValue=""});
window.addEventListener("popstate",()=>{const requested=location.hash.replace(/^#/,"")||"home";if($(`.view#${CSS.escape(requested)}`))go(requested,{historyMode:"none"})});
initialize();pollJob();setInterval(pollJob,1000);pollGpu();setInterval(pollGpu,2500);
setInterval(()=>{if($("#samples").classList.contains("active")&&state.activeJob&&["starting","running"].includes(state.activeJob.status))loadSamples().catch(()=>{})},5000);
