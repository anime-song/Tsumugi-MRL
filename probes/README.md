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

## Mel-RVQ pretraining

The encoder was pretrained for 625 epochs with `--ablation mel_rvq` on 17,037
audio/MIDI pairs and exported to `checkpoints/pretraining_mel/audio_model`.
Probe settings match the random baseline above, so only the frozen encoder
differs.

| Probe | Best / max epoch | Test metrics |
| --- | ---: | --- |
| NSynth pitch | 19 / 30 | accuracy 85.94%, macro-F1 71.99% |
| NSynth instrument family | 11 / 30 | accuracy 67.14%, macro-F1 61.92% |
| GuitarSet chord | 2 / 30 | accuracy 40.09%, macro-F1 31.50% |
| FMA-small genre | 3 / 100 | accuracy 45.50%, macro-F1 22.40% |
| OpenMIC instrument | 99 / 100 | micro-F1 42.01%, macro-F1 41.25% |
| Ballroom beat | 8 / 30 | F1 78.87%, accuracy 88.69% |
| GiantSteps key | 6 / 30 | accuracy 41.38%, macro-F1 23.37% |

Acoustic pretraining raises every probe except GiantSteps key, where macro-F1
drops from 27.35% to 23.37%. Beat tracking gains the most (+29.53 points of
F1), followed by instrument recognition on OpenMIC and NSynth. Results are
saved under each probe's `results/mel_rvq.json`, and NSynth instrument family
uses `results/family_mel_rvq.json`.
