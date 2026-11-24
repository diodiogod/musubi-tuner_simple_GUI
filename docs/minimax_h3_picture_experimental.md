# MiniMax H3 Image LoRA

Choose **MiniMax H3 Image (Experimental)** at the bottom of the model list in Modern or Classic. Use captioned still images at batch size 1. Gradient accumulation can combine multiple images into each optimizer update. Build both caches for the first run with this model.

Select these files in the model setup.

- **H3 Image DiT** uses `minimax_h3_image_pruned_int8_convrot.safetensors`.
- **H3 Image VAE** uses `minimax_h3_image_vae_fp16.safetensors`.
- **Qwen3-VL Text Encoder** can reuse the shared MiniMax H3 Qwen3-VL-32B checkpoint in ConvRot INT8 or NVFP4 AWQ. [Model files](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main)
- **H3 Image Training Adapter** is optional. A matching Image adapter stays frozen during training and switches off during previews. [Fizgig H3 Image training adapter](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter)

Training samples the checkpoint’s six native noise levels. Preview uses the released six-step schedule, guidance 1, and a single output image. Set image dimensions in multiples of 32 and leave the negative prompt empty. This first integration covers text-to-image LoRA with standard captioned images. It has been checked with synthetic tensors and command parsing.

The six-head DiT and 2D VAE behavior were checked against the [AI Toolkit H3 Image implementation](https://github.com/ostris/ai-toolkit/commit/5b1670a5b8c411e71c6d32a704e873dee4e74a9f) and [ComfyUI H3 model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py). [Fizgig’s H3 Image guide](https://github.com/shootthesound/Fizgig/blob/master/docs/MINIMAX_IMAGE.md) describes its own training recipes and measurements.
