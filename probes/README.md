# Linear probes

The probes freeze `TsumugiMRLModel` and train a linear head on the mean-pooled
audio representation.

## Download

```powershell
uv run --extra probe python scripts/download_probe_datasets.py
```

## Run

Use `--checkpoint` for a pretrained encoder:

```powershell
uv run --extra probe python -m probes.nsynth.probe --task pitch --checkpoint checkpoints/pretraining/audio_model
uv run --extra probe python -m probes.nsynth.probe --task family --checkpoint checkpoints/pretraining/audio_model
uv run --extra probe python -m probes.guitarset.probe --checkpoint checkpoints/pretraining/audio_model
uv run --extra probe python -m probes.fma_small.probe --checkpoint checkpoints/pretraining/audio_model
uv run --extra probe python -m probes.openmic.probe --checkpoint checkpoints/pretraining/audio_model
uv run --extra probe python -m probes.ballroom.probe --checkpoint checkpoints/pretraining/audio_model
uv run --extra probe python -m probes.giantsteps_key.probe --checkpoint checkpoints/pretraining/audio_model
```

Use `--random-init` for a frozen randomly initialized encoder:

```powershell
uv run --extra probe python -X utf8 -m probes.nsynth.probe --task pitch --random-init --seed 0 --feature-cache probes/cache/nsynth_random_seed0.pt
```

`--feature-cache` stores extracted features on disk and avoids repeating audio
decoding and encoder inference. The linear head still loads the cached matrix
into RAM. Use `--max-items 100` for a smoke run and `--device cuda` to select
the GPU explicitly.

GuitarSet and Ballroom are frame-level probes using the first 30 seconds of
each recording.

## Random baseline

All results below use seed 0, patience-10 early stopping, and head batch size
8192. The maximum epoch count is dataset-specific:

| Dataset | Max epochs |
| --- | ---: |
| NSynth pitch/family | 30 |
| GuitarSet | 30 |
| Ballroom | 30 |
| GiantSteps Key | 30 |
| FMA-small | 100 |
| OpenMIC | 100 |

OpenMIC has no official validation split, so 10% of its training data is used
for deterministic early stopping. FMA-small excludes the three non-decodable
tracks `099134`, `108925`, and `133297`, leaving 7,997 tracks.

| Probe | Best / max epoch | Test metrics |
| --- | ---: | --- |
| NSynth pitch | 25 / 30 | accuracy 82.71%, macro-F1 68.22% |
| NSynth instrument family | 28 / 30 | accuracy 47.29%, macro-F1 43.19% |
| GuitarSet chord | 30 / 30 | accuracy 27.78%, macro-F1 15.53% |
| FMA-small genre | 100 / 100 | accuracy 36.88%, macro-F1 17.65% |
| OpenMIC instrument | 52 / 100 | micro-F1 25.89%, macro-F1 25.45% |
| Ballroom beat | 30 / 30 | F1 49.33%, accuracy 66.89% |
| GiantSteps key | 18 / 30 | accuracy 48.28%, macro-F1 27.35% |

Results are saved under each probe's `results/random.json`. Training details
such as `epochs_run`, `best_epoch`, and the validation source are included in
the `training` field.
