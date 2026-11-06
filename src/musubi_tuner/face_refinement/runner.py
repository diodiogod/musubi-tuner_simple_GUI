"""Shared DRaFT face-refinement loop for supported model families."""
import json
from pathlib import Path
import random
import torch

from musubi_tuner.face_refinement.draft import reward_loss, save_preview
from musubi_tuner.face_refinement.face_reward import FaceSimilarityReward
from musubi_tuner.face_refinement.pose import parse_pose_prompt


def run_face_loop(args, model, vae, network, parameters, optimizer, embeddings, prompts,
                  prompt_records, pose_plan, pose_tracker, device, generate_fn, save_fn):
    vae.to(device)
    reference_images = args.reference_dir
    pose_buckets = {}
    if args.reference_manifest:
        manifest = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
        reference_images = manifest.get("reference_images", manifest) if isinstance(manifest, dict) else manifest
        if reference_images and isinstance(reference_images[0], dict):
            pose_buckets = {str(item["path"]): str(item.get("pose", "uncertain")) for item in reference_images}
            reference_images = [str(item["path"]) for item in reference_images if item.get("enabled", True)]
        if not reference_images:
            raise ValueError("The reference manifest contains no enabled face images")
    pose_weight = args.pose_reward_weight if args.pose_aware else 0.0
    pose_targets = {}
    if pose_tracker:
        pose_weight = min(pose_weight, max(0.0, 1.0 - float(pose_plan.get("overall_anchor_weight", 0.80))))
        pose_targets = {pose: float(cfg.get("target", args.target_similarity)) for pose, cfg in pose_plan.get("buckets", {}).items()}
    reward = FaceSimilarityReward(
        reference_images=reference_images,
        model_dir=args.face_model_dir,
        target_similarity=args.target_similarity,
        reference_entropy_weight=args.anti_copy_weight,
        expression_diversity_weight=0.0,
        pose_buckets=pose_buckets if args.pose_aware else None,
        pose_reward_weight=pose_weight,
        pose_min_references=args.pose_min_references,
        pose_targets=pose_targets,
        providers=["CPUExecutionProvider"],
        device=device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    consecutive_target = 0
    detected_steps = 0
    randomizer = random.Random(args.seed)
    prompt_weights = [float(item.get("weight", 1.0)) for item in prompt_records] if len(prompt_records) == len(prompts) else None
    metrics_path = output.parent / "face_refinement_pose_metrics.jsonl"
    for step in range(1, args.train_steps + 1):
        prompt_index = randomizer.choices(range(len(prompts)), weights=prompt_weights, k=1)[0] if prompt_weights else randomizer.randrange(len(prompts))
        prompt = prompts[prompt_index]
        text, text_mask, negative, negative_mask = embeddings[prompt_index]
        optimizer.zero_grad(set_to_none=True)
        pixels = generate_fn(
            model, vae, text, text_mask, negative, negative_mask,
            resolution=args.resolution, denoise_steps=args.denoise_steps,
            draft_k=args.draft_k, cfg_scale=args.cfg_scale,
            seed=args.seed + step, checkpoint_vae=args.checkpoint_vae,
        )
        loss, reward_value = reward_loss(reward, pixels, prompt)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        metrics = reward.last_metrics
        detected_steps += int(metrics.get("face_detected", 0))
        similarity = metrics.get("face_similarity", 0.0)
        consecutive_target = consecutive_target + 1 if similarity >= args.stop_similarity else 0
        print(
            f"step={step}/{args.train_steps} loss={loss.item():.6f} reward={reward_value.item():.6f} "
            f"face_similarity={similarity:.4f} face_detected={int(metrics.get('face_detected', 0))} "
            f"grad_norm={float(grad_norm):.4f}", flush=True,
        )
        if pose_tracker:
            pose_bucket = str(metrics.get("pose_bucket") or parse_pose_prompt(prompt)[0] or "")
            pose_similarity = float(metrics.get("pose_similarity", 0.0))
            if pose_bucket:
                pose_tracker.update(pose_bucket, pose_similarity)
                status = pose_tracker.pose_status(pose_bucket)
                record = {"step": step, "pose": pose_bucket, "similarity": pose_similarity, **status}
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"pose_metric={pose_bucket} similarity={pose_similarity:.4f} "
                    f"evaluations={status['evaluations']} target={pose_plan['buckets'][pose_bucket]['target']}",
                    flush=True,
                )
        if args.preview_every > 0 and (step == 1 or step % args.preview_every == 0):
            preview = save_preview(pixels, output.parent, step, prompt)
            print(f"face_preview={preview}", flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            save_fn(network, output.with_name(f"{output.stem}-{step:06d}{output.suffix}"), args, step)
        if step >= args.min_steps and detected_steps / step < args.min_detection_rate:
            raise RuntimeError("Face detection rate fell below the configured safety threshold")
        pose_stop_reason = pose_tracker.stop_reason() if pose_tracker else None
        if pose_stop_reason:
            print(f"early_stop={pose_stop_reason}", flush=True)
            summary = {pose: pose_tracker.pose_status(pose) for pose, cfg in pose_plan["buckets"].items() if cfg.get("enabled")}
            (output.parent / "face_refinement_pose_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            save_fn(network, output, args, step)
            return
        if not pose_tracker and consecutive_target >= args.early_stop_patience:
            print(f"early_stop=target_similarity_reached patience={consecutive_target}", flush=True)
            save_fn(network, output, args, step)
            return
    if pose_tracker:
        summary = {pose: pose_tracker.pose_status(pose) for pose, cfg in pose_plan["buckets"].items() if cfg.get("enabled")}
        (output.parent / "face_refinement_pose_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("pose_stop=max_steps_reached", flush=True)
    save_fn(network, output, args, args.train_steps)
