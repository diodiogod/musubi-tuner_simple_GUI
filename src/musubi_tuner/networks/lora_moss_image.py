"""MOSS adapters over main-block attention and feed-forward projections.

Keep the frozen text refiner, shared modulation and input/output projections
unchanged, matching the standard Kohya MOSS training scope.
"""
from musubi_tuner.networks import lora

TARGET_MODULES = ["MossImageBlock"]


def create_arch_network(multiplier, network_dim, network_alpha, vae,
                        text_encoders, unet, neuron_dropout=None, **kwargs):
    module_class = None
    if str(kwargs.pop("moss_lowp_lora", False)).lower() == "true":
        from musubi_tuner.moss_image.fast_lora import FastMossLoRA
        module_class = FastMossLoRA
    return lora.create_network(
        TARGET_MODULES, "lora_unet", multiplier, network_dim, network_alpha,
        vae, text_encoders, unet, neuron_dropout=neuron_dropout, module_class=module_class, **kwargs,
    )


def create_arch_network_from_weights(multiplier, weights_sd, text_encoders=None,
                                     unet=None, for_inference=False, **kwargs):
    module_class = None
    if str(kwargs.pop("moss_lowp_lora", False)).lower() == "true" and not for_inference:
        from musubi_tuner.moss_image.fast_lora import FastMossLoRA
        module_class = FastMossLoRA
    return lora.create_network_from_weights(
        TARGET_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, module_class=module_class, **kwargs,
    )
