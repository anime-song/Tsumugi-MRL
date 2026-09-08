# FMA-small genre probe

This probe uses the official FMA-small train/validation/test split and the
16 top-level genres. Audio is represented by a mean-pooled frozen
Tsumugi-MRL embedding from the first 30 seconds.

```powershell
uv run --extra probe python -m probes.fma_small.probe --checkpoint checkpoints/pretraining/audio_model
```

The official FMA-small archive contains three known non-decodable MP3 files:
`099134`, `108925`, and `133297`. The probe excludes these files explicitly
and records the IDs in the result JSON, so the random baseline uses 7,997
tracks.

The default linear-head budget is 100 epochs with patience-10 early stopping
on the official validation split.
