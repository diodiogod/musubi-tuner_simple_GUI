# Third-Party Notices

## Krea 2 DRaFT face reward

`src/musubi_tuner/face_refinement/face_reward.py` is adapted from
[KONAKONA666/krea-2](https://github.com/KONAKONA666/krea-2), retrieved on
2026-07-11. The upstream source is offered under the Apache License 2.0; a copy
is included at `licenses/Apache-2.0.txt`.

Local modifications integrate the reward with Musubi, add step metrics, and
apply local package/lint conventions. The DRaFT method itself is described in
[Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400).

The optional AntelopeV2/InsightFace model files are third-party artifacts with
separate terms. They are downloaded only after an explicit user action and are
not included in this repository. Users must review their terms before use,
especially for commercial purposes.
