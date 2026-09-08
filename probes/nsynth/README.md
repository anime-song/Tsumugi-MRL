# NSynth probes

NSynth provides the cleanest low-level probe in this suite: 112-way pitch and
instrument-family classification on 4-second monophonic notes. The official
split keeps instrument identities disjoint, so the result is not merely a
memorization test of instrument instances.

```powershell
uv run --extra probe python -m probes.nsynth.probe --task pitch --checkpoint checkpoints/pretraining/audio_model
uv run --extra probe python -m probes.nsynth.probe --task family --checkpoint checkpoints/pretraining/audio_model
```
