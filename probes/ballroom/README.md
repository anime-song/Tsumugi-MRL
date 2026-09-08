# Ballroom beat probe

Ballroom supplies independent beat annotations, which makes it a useful
rhythmic probe alongside the symbolic/audio pretraining task. The script
aligns annotated beat times to the model's 25-Hz hidden frames and reports
frame-level F1. Because the public annotations document duplicate recordings,
review the resulting track list before treating this as a strict
cross-recording generalization score.

```powershell
uv run --extra probe python -m probes.ballroom.probe --checkpoint checkpoints/pretraining/audio_model
```
