# GiantSteps Key probe

GiantSteps Key is the song-level harmonic probe: 24 major/minor key classes
from Beatport previews. Since the dataset does not provide a train/validation/
test split for this use, the script creates a deterministic hash split by
track ID. For a publication result, replace it with an artist-aware split if
additional metadata is available.

```powershell
uv run --extra probe python -m probes.giantsteps_key.probe --checkpoint checkpoints/pretraining/audio_model
```
