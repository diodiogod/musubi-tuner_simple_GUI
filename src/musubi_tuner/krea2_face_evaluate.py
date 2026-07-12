"""CLI for scoring a fixed Krea Turbo face evaluation suite."""

import argparse
import json

from musubi_tuner.face_refinement.evaluation import evaluate_suite


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated Krea Turbo faces without training")
    for name in ("suite", "images_dir", "reference_manifest", "face_model_dir", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    result = evaluate_suite(args.suite, args.images_dir, args.reference_manifest, args.face_model_dir, args.output, args.baseline)
    for pose, metrics in result["poses"].items():
        print(f"turbo_eval={pose} identity={metrics['overall_similarity']} pose={metrics['pose_similarity']} pose_success={metrics['pose_success_rate']:.3f} detection={metrics['detection_rate']:.3f}", flush=True)
    print(f"turbo_eval_result={args.output}", flush=True)


if __name__ == "__main__": main()
