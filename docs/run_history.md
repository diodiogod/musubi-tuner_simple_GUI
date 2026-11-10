# Reviewing training runs

Modern UI **History** shows Classic and Modern jobs together. It starts with
training runs; choose **All jobs** to include standalone preview jobs. Existing
continuation, exact recovery, recipe replay, speed comparisons and lineage views
remain available.

Each training row keeps its compact height, with a small loss curve beside the
speed. Click the curve, or choose **Open run review** in the row menu, to access
the following tabs:

- **Loss graph** opens the same chart used by **Active run**. Adjust smoothing,
  then use **Display options** for raw loss, the trainer's running average, epoch
  means, learning rate, log scale and outlier clipping. Hover, or focus the graph
  and use arrow keys, to inspect steps. The legend identifies visible series.
- **Previews & checkpoints** shows files currently in the recorded output folder.
  Click an image or video to open it at full size. A folder reused by multiple
  attempts contains their shared outputs; moved or removed files cannot appear.
- **Notes** saves observations for that specific history entry. Notes live in
  ignored `job_notes_local.json`, separately from the training recipe and the LoRA
  training comment. They remain attached when history ordering changes.

## What the chart actually knows

Older jobs usually recorded `avr_loss`, the trainer's running average. The UI
labels those curves accordingly and disables missing raw-loss/epoch/LR options.
Where available, saved console logs restore the curve lazily. Files are never
modified to recover chart data. The managed log reader limits large logs to their
latest 8 MB; the chart reports this limitation.

New shared-trainer runs also put the already-computed current loss and first
scheduler learning rate into the progress output. This adds no GPU operations or
training passes. Captured updates can omit intermediate steps; displayed epoch
means use the raw updates that were captured, rather than estimating missing
values. Learning rate uses its own right-hand scale. Smoothing and outlier
clipping change only the display. A low loss is not a sample-quality score.

The compact progress and optional regularization metrics above the live chart
keep their existing layout. More detailed analysis belongs in the chart and run
review window.

## Design reference

The run-review actions, compact summary curves and chart controls were informed
by the installed Apache-2.0 [Fizgig](https://github.com/shootthesound/Fizgig)
frontend at commit `fe1fb5d26088b162fe7ab67a045b0ec88bc7f25d`
(`src/fizgig/web/static/app.js`, run history and loss graph). The local
implementation is separate and accommodates Musubi's existing history/log data;
it does not copy Fizgig's training or history persistence layer.
