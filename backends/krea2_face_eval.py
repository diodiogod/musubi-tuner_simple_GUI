"""Prepare fixed Turbo generation/evaluation commands for face refinement."""

import json
from datetime import datetime
from pathlib import Path

from musubi_tuner.face_refinement.lora_validation import render_trigger_prompts
from musubi_tuner.face_refinement.pose import parse_pose_prompt
from musubi_tuner.face_refinement.pose_plan import weighted_prompt_records


def prepare(settings, config, input_lora, *, baseline_result=None, label="baseline"):
    if not settings.get("krea2_turbo_dit") or not Path(settings["krea2_turbo_dit"]).is_file():
        raise ValueError("Turbo evaluation requires an existing Krea 2 Turbo DiT model path.")
    if not input_lora or not Path(input_lora).is_file():
        raise ValueError("Choose an existing Krea 2 LoRA to evaluate.")
    plan = config.get("pose_plan") or {}
    records = weighted_prompt_records(plan) if config.get("pose_aware") and plan.get("enabled") else []
    if not records:
        records = []
        for prompt in config.get("prompts", []):
            pose, _ = parse_pose_prompt(prompt)
            if pose and pose != "auto": records.append({"pose": pose, "prompt": prompt, "weight": 1})
    if not records:
        raise ValueError("Turbo pose evaluation needs at least one explicitly pose-tagged prompt or an enabled Pose Training Plan.")

    prompts_per_pose = max(1, int(config.get("evaluation_prompts_per_pose", 1)))
    seeds_per_prompt = max(1, int(config.get("evaluation_seeds_per_prompt", 2)))
    base_seed = int(config.get("evaluation_seed", 42000))
    grouped = {}
    for record in records: grouped.setdefault(record["pose"], []).append(record["prompt"])
    cases = []; prompt_lines = []; case_index = 0
    resolution = int(config.get("evaluation_resolution", 512)); steps = int(config.get("evaluation_steps", 8))
    for pose, prompts in grouped.items():
        for prompt in prompts[:prompts_per_pose]:
            rendered = render_trigger_prompts([prompt], config.get("trigger_word", ""))[0]
            _, clean = parse_pose_prompt(rendered)
            for seed_offset in range(seeds_per_prompt):
                seed = base_seed + case_index * 100 + seed_offset
                cases.append({"id": f"{pose}-{case_index}-{seed_offset}", "pose": pose, "prompt": clean, "seed": seed})
                prompt_lines.append(f"{clean} --w {resolution} --h {resolution} --s {steps} --g 1.0 --d {seed}")
            case_index += 1

    projector_strength = float(settings.get("krea2_projector_diff_strength") or 1.0)
    renderer_settings = {
        "turbo_dit": str(Path(settings["krea2_turbo_dit"]).resolve()),
        "vae": str(Path(settings["vae_model"]).resolve()),
        "text_encoder": str(Path(settings["krea2_text_encoder"]).resolve()),
        "projector_diff": str(Path(settings["krea2_projector_diff"]).resolve()) if settings.get("krea2_projector_diff") else "",
        "projector_strength": projector_strength,
        "resolution": resolution, "steps": steps, "guidance": 1.0,
        "attention": settings.get("attention_mechanism", "sdpa"), "fp8_scaled": bool(settings.get("fp8_scaled")),
        "split_attn": bool(settings.get("split_attn")), "blocks_to_swap": int(settings.get("blocks_to_swap") or 0),
    }
    if baseline_result:
        baseline_payload = json.loads(Path(baseline_result).read_text(encoding="utf-8"))
        baseline_suite_path = Path(baseline_payload.get("suite", ""))
        if not baseline_suite_path.is_file(): raise ValueError("The baseline result does not point to its original fixed evaluation suite.")
        baseline_suite = json.loads(baseline_suite_path.read_text(encoding="utf-8"))
        expected = baseline_suite.get("renderer_settings", {})
        for key in ("turbo_dit", "vae", "text_encoder", "projector_diff", "projector_strength", "attention", "fp8_scaled", "split_attn", "blocks_to_swap"):
            if expected.get(key) != renderer_settings.get(key):
                raise ValueError(f"Comparison setting '{key}' differs from the baseline. Restore the baseline Turbo/projector settings for a fair comparison.")
        cases = baseline_suite["cases"]; resolution = int(expected.get("resolution", resolution)); steps = int(expected.get("steps", steps))
        renderer_settings.update({"resolution": resolution, "steps": steps})
        prompt_lines = [f"{case['prompt']} --w {resolution} --h {resolution} --s {steps} --g 1.0 --d {int(case['seed'])}" for case in cases]

    root = Path(settings["output_dir"]) / (settings.get("output_name") or "face_refinement") / "face_evaluations"
    run_dir = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{label}"
    images_dir = run_dir / "images"; images_dir.mkdir(parents=True, exist_ok=True)
    suite_path = run_dir / "suite.json"; prompts_path = run_dir / "turbo_prompts.txt"
    result_path = run_dir / "results.json"; refs_path = run_dir / "references.json"
    suite_path.write_text(json.dumps({"renderer": "krea2_turbo", "renderer_settings": renderer_settings, "pose_min_references": config.get("pose_min_references", 2), "cases": cases}, indent=2), encoding="utf-8")
    prompts_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    excluded = set(config.get("excluded_reference_images") or [])
    reference_entries = [{"path": item["path"], "pose": item.get("bucket", "uncertain"), "enabled": item["path"] not in excluded}
                         for item in (config.get("preflight_report") or {}).get("scored_images", [])]
    refs_path.write_text(json.dumps({"reference_images": reference_entries}, indent=2), encoding="utf-8")

    python = settings.get("python_executable") or "python"
    attention = {"sdpa": "torch", "none": "torch", "flash_attn": "flash", "sage_attn": "sageattn"}.get(settings.get("attention_mechanism"), settings.get("attention_mechanism", "torch"))
    generate = [python, "src/musubi_tuner/krea2_generate_image.py", "--dit", settings["krea2_turbo_dit"], "--vae", settings["vae_model"], "--text_encoder", settings["krea2_text_encoder"], "--save_path", str(images_dir), "--attn_mode", attention, "--turbo", "--from_file", str(prompts_path), "--lora_weight", str(input_lora)]
    if settings.get("fp8_scaled"): generate.append("--fp8_scaled")
    if settings.get("split_attn"): generate.append("--split_attn")
    if settings.get("blocks_to_swap"): generate.extend(["--blocks_to_swap", str(settings["blocks_to_swap"])])
    if settings.get("krea2_projector_diff"): generate.extend(["--projector_diff", settings["krea2_projector_diff"], "--projector_diff_strength", str(projector_strength)])
    evaluate = [python, "src/musubi_tuner/krea2_face_evaluate.py", "--suite", str(suite_path), "--images_dir", str(images_dir), "--reference_manifest", str(refs_path), "--face_model_dir", str(config["face_model_dir"]), "--output", str(result_path)]
    if baseline_result: evaluate.extend(["--baseline", str(baseline_result)])
    return {"commands": [generate, evaluate], "run_dir": run_dir, "result": result_path, "cases": len(cases)}
