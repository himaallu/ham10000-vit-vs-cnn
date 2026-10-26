# Run logs

Copied verbatim from the Colab output (the notebook prints them; the files on Google Drive are the source):

- `lr_check_output.txt`: section 5 of the notebook (3-epoch learning-rate check, train/val only).
- `run_times_output.txt`: GPU name (section 2) and minutes per run = time between `config.yaml`
  (written when a run starts) and `summary.json` (written when it ends). Run 1 seed 0 shows 0
  because it was re-run after finishing, which rewrote `config.yaml`; it is ignored.
