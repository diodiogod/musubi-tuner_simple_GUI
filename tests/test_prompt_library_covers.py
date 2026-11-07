from PIL import Image
from prompt_library import PromptLibraryStore
from preview_profiles import prompt_defaults, preview_metadata, starter_prompts


def test_covers_follow_checkpoint_and_keep_other_models(tmp_path):
    store = PromptLibraryStore(tmp_path / "library")
    image = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), "red").save(image)
    entry, _ = store.add_or_merge({"prompt": "a teapot", "steps": 50})
    prompt = {"prompt": "a teapot", "steps": 25, "_library_id": entry["id"]}
    moss = {"training_mode": "MOSS-Image (Experimental)", "moss_dit_model": str(tmp_path / "base")}
    h3 = {"training_mode": "MiniMax H3 (Experimental)", "minimax_h3_dit_model": str(tmp_path / "h3")}
    for settings in (moss, h3):
        store.capture_thumbnail(prompt, image, settings["training_mode"], preview_metadata(settings))
    # Repeated refreshes of MOSS may not evict H3's cover.
    for _ in range(8):
        store.capture_thumbnail(prompt, image, moss["training_mode"], preview_metadata(moss))
    assert len(store.prompts) == 1
    assert entry["prompt_data"]["steps"] == 50
    assert len(entry["thumbnails"]) == 5
    for settings in (moss, h3):
        meta = preview_metadata(settings)
        selected = store.latest_thumbnail(entry, meta["mode"], meta["model_key"])
        assert selected["metadata"]["model_key"] == meta["model_key"]
    store.capture_thumbnail(prompt, image, moss["training_mode"], preview_metadata(moss, "trained.safetensors"))
    assert store.latest_thumbnail(entry, moss["training_mode"], preview_metadata(moss)["model_key"])["metadata"]["baseline"]
    reloaded = PromptLibraryStore(store.root)
    assert reloaded.latest_thumbnail(reloaded.prompts[0], h3["training_mode"])["mode"] == h3["training_mode"]
    assert reloaded.latest_thumbnail(reloaded.prompts[0], "another family") is not None


def test_changed_text_does_not_replace_original_library_cover(tmp_path):
    store = PromptLibraryStore(tmp_path / "library")
    image = tmp_path / "sample.png"
    Image.new("RGB", (32, 32)).save(image)
    entry, _ = store.add_or_merge({"prompt": "a teapot"})
    store.capture_thumbnail({"prompt": "a person", "_library_id": entry["id"]}, image, "MOSS")
    assert not entry["thumbnails"]
    assert len(store.prompts) == 2


def test_preview_metadata_does_not_duplicate_prompt_and_starters_are_portable(tmp_path):
    store = PromptLibraryStore(tmp_path)
    for prompt in starter_prompts():
        entry, created = store.add_or_merge(prompt)
        assert created and entry["name"] != "Untitled prompt"
        assert not any(key.startswith("_") for key in entry["prompt_data"])
        assert not store.add_or_merge(dict(prompt, _preview_use_lora=False))[1]
    defaults = prompt_defaults({"training_mode": "MOSS-Image (Experimental)"})
    assert (defaults["steps"], defaults["sampler"], defaults["scheduler"]) == (25, "ddim", "simple")


def test_video_preview_produces_a_viewable_library_cover(tmp_path):
    import av
    video = tmp_path / "sample.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), "red"))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    store = PromptLibraryStore(tmp_path / "library")
    entry, _ = store.capture_thumbnail({"prompt": "a teapot"}, video, "MiniMax H3 (Experimental)")
    with Image.open(store.thumbnail_path(store.latest_thumbnail(entry))) as poster:
        assert poster.size == (32, 32)
        assert poster.getpixel((16, 16))[0] > 200
