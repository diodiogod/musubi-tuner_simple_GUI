# Browsing and comparing previews

Modern UI **Samples** starts in a full-width gallery for the current run. Nothing
is generated when you browse these views.

- **Prompt progression** shows prompts down the left and checkpoints from left
  to right. Epoch and step previews have separate grids. Empty cells mean no
  preview was saved for that checkpoint. Repeated takes remain available.
- **Epoch / step** groups the prompts generated at each checkpoint together.
- **None · newest first** shows the flat gallery. **Size** changes thumbnail size.

Click a thumbnail for its full preview. In **Prompt progression**, the preview's
arrow buttons follow that prompt across checkpoints. In the flat gallery, they
follow the gallery order. **Details** shows the prompt, seed, source, filename,
dimensions and available sampling settings.

To compare, click **Select to compare**, pick reference **A**, pick **B**, and
click **Compare 2**. A third selection replaces B. Image and video selections
cannot be paired with each other.

The comparison viewer keeps **A pinned** by default. Arrows advance only B.
Click **Reference A** or **Compare B** before choosing a thumbnail to change that
side. Turn off **A pinned** to compare neighboring checkpoints as B advances.
The existing wipe, side-by-side, zoom, full-preview and synchronized-video tools
remain available. The Refresh button preserves selections by file identity.

**This run only** is the default scope. Previously added folders stay saved but
are not mixed into the current run automatically. Choose **This run + selected
folders**, then check the folders to include in **Folders…**. This selection is
separate from the saved folder list; removing a check does not delete its files
or remove the saved folder. Switching the training output resets to the current
run. Comparison remains possible across explicitly selected folders.

Where image metadata is unavailable, prompt descriptions come from saved
training-job settings matching the output and generation time. The details panel
labels this as **Saved run settings**. Current workspace prompts are not applied
to old previews. Missing descriptions stay as numbered prompt labels. Different
known captions, seeds, image dimensions and media formats are kept as separate series;
manual A/B selection can still compare different image series. Differences in
known prompts, seeds or dimensions are shown in the comparison viewer.
