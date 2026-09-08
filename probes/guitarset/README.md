# GuitarSet chord probe

This probe uses the `mic` mix and JAMS chord annotations. Audio is segmented
into the model's approximately 25 Hz frame states, and a single linear head
predicts the chord class at each frame. Splits are by player ID to avoid
putting the same guitarist in both train and test.

```powershell
uv run --extra probe python -m probes.guitarset.probe --checkpoint checkpoints/pretraining/audio_model
```
