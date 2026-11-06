from pathlib import Path
import torch
from test_minimax_h3_native_text_encoder import _record, _visual
from musubi_tuner.minimax_h3_native.media import H3Reference
from musubi_tuner.minimax_h3_native.text_encoder import (
    build_presentation, build_token_tags, VISION_START_TOKEN_ID, VISION_END_TOKEN_ID,
    IMAGE_TOKEN_ID, VIDEO_TOKEN_ID,
)

def _offline_qwen3_vl_processor():
    # a real Qwen3VLProcessor over a word-level tokenizer that knows only the vision tokens (at the
    # released ids), so the placeholder expansion runs exactly as with the released processor
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    special = {
        "<|vision_start|>": VISION_START_TOKEN_ID,
        "<|vision_end|>": VISION_END_TOKEN_ID,
        "<|image_pad|>": IMAGE_TOKEN_ID,
        "<|video_pad|>": VIDEO_TOKEN_ID,
    }
    word_level = Tokenizer(models.WordLevel({"<unk>": 0, **special}, unk_token="<unk>"))
    word_level.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=word_level, unk_token="<unk>")
    tokenizer.add_special_tokens({"additional_special_tokens": list(special)})
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )
    return processor, tokenizer


def test_real_processor_expands_a_video_reference_into_the_presentation_text(tmp_path: Path):
    # the processor expands a bare <|video_pad|> into one "<t seconds><|vision_start|>...<|vision_end|>"
    # span per two-frame block; a wrapped placeholder used to leave its own vision-start/end around
    # those spans, which build_token_tags rejected as nested vision-start tokens
    processor, tokenizer = _offline_qwen3_vl_processor()
    image = tmp_path / "face.png"
    video = tmp_path / "motion.mp4"
    record = _record(
        tmp_path,
        (
            H3Reference(type="image", path=image),
            H3Reference(type="video", path=video, audio=None, duration_seconds=2.0),
        ),
    )
    presentation = build_presentation(
        record,
        "ref2va",
        {image: _visual(1), video: _visual(4, [0.0, 0.5, 1.0, 1.5])},
    )

    processed = processor(
        text=[presentation.processor_text],
        images=[frame.numpy() for frame in presentation.images],
        videos=[frames.numpy() for frames in presentation.videos],
        video_metadata=[
            {
                "total_num_frames": 4,
                "fps": 2.0,
                "duration": 2.0,
                "frames_indices": [0, 1, 2, 3],
                "height": 32,
                "width": 32,
            }
        ],
        do_sample_frames=False,
        return_tensors="pt",
        padding=False,
    )
    tags = build_token_tags(processed)

    input_ids = processed["input_ids"][0].tolist()
    assert input_ids.count(VISION_START_TOKEN_ID) == 3  # one picture + two video blocks
    assert int((tags == 0).sum()) == sum(
        token in (VISION_START_TOKEN_ID, VISION_END_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID) for token in input_ids
    )
    # with each run of expanded rows collapsed back to one placeholder, the processor output is the
    # presentation text, block timestamps included
    collapsed = [
        token
        for index, token in enumerate(input_ids)
        if token not in (IMAGE_TOKEN_ID, VIDEO_TOKEN_ID) or index == 0 or input_ids[index - 1] != token
    ]
    assert collapsed == tokenizer(presentation.text)["input_ids"]


