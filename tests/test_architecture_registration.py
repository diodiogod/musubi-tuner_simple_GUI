"""Every declared training architecture must support caches and adapter saving.

This catches a common integration failure when a new trainer works in isolation
but is absent from shared dataset or output-metadata registries.
"""
import pytest
from musubi_tuner.dataset import architectures
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.utils.sai_model_spec import build_metadata


@pytest.mark.parametrize("name,architecture", sorted(
    (name, value) for name, value in vars(architectures).items()
    if name.startswith("ARCHITECTURE_") and not name.endswith("_FULL")
))
def test_training_architectures_have_bucket_and_save_metadata(name, architecture):
    assert architecture in BucketSelector.ARCHITECTURE_STEPS_MAP, f"{name} has no bucket alignment"
    metadata = build_metadata(None, architecture, timestamp=0, is_lora=True)
    assert metadata["modelspec.architecture"].endswith("/lora")
    assert metadata["modelspec.implementation"]
