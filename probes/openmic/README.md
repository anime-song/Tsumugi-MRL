# OpenMIC probe

OpenMIC is the complementary instrument probe to NSynth: it is multi-label
and polyphonic rather than a single isolated note. The probe uses the official
partition and reports micro/macro F1 from a frozen clip embedding.

```powershell
uv run --extra probe python -m probes.openmic.probe --checkpoint checkpoints/pretraining/audio_model
```

The default linear-head budget is 100 epochs with patience-10 early stopping.
Because OpenMIC has no official validation split, 10% of the training data is
held out deterministically for early stopping.
