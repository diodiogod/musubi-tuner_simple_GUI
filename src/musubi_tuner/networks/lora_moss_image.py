"""MOSS adapters over main-block attention and feed-forward projections.

Keep the frozen text refiner, shared modulation and input/output projections
unchanged, matching the standard Kohya MOSS training scope.
"""
from musubi_tuner.networks import lora

TARGET_MODULES = ["MossImageBlock"]


def create_arch_network(multiplier, network_dim, network_alpha, vae,
                        text_encoders, unet, neuron_dropout=None, **kwargs):
    return lora.create_network(
        TARGET_MODULES, "lora_unet", multiplier, network_dim, network_alpha,
        vae, text_encoders, unet, neuron_dropout=neuron_dropout, **kwargs,
    )


def create_arch_network_from_weights(multiplier, weights_sd, text_encoders=None,
                                     unet=None, for_inference=False, **kwargs):
    return lora.create_network_from_weights(
        TARGET_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs,
    )
