"""MOSS specialization of the shared DRaFT face-stage command builder."""
from backends import krea2_face
from backends._common import add_arg


def build_command(settings, config, input_lora, output_lora, prompts_json):
    mapped = dict(settings, krea2_dit_model=settings.get("moss_dit_model"),
                  krea2_text_encoder=settings.get("moss_text_encoder"), fp8_scaled=False,
                  krea2_projector_diff=None, krea2_projector_diff_strength=None)
    command = krea2_face.build_command(mapped, config, input_lora, output_lora, prompts_json)
    command[1] = "src/musubi_tuner/moss_image_face_refinement.py"
    for flag, key in (("tokenizer_cache_dir", "moss_tokenizer_cache_dir"),
                      ("moss_max_token_length", "moss_max_token_length"),
                      ("moss_sample_flow_shift", "moss_sample_flow_shift"),
                      ("convrot_bwd_mode", "moss_convrot_bwd_mode")):
        add_arg(command, "--" + flag, settings.get(key))
    command.extend(["--moss_sample_skip_layers", str(settings.get("moss_sample_skip_layers", "22"))])
    add_arg(command, "--text_encoder_cpu_streaming", settings.get("moss_text_encoder_cpu_streaming", True))
    add_arg(command, "--convrot_int8", settings.get("moss_convrot_int8", False))
    return command


output_path = krea2_face.output_path
