# Symbolic teacher training

Train a MIDI teacher to produce symbolic targets for Tsumugi-MRL pretraining.

## Prepare data

Run from the repository root, setting `--amt-root` to your
`instrument_agnostic_amt` checkout:

```powershell
uv run --extra train python -m train.symbolic_teacher.prepare_dataset `
  --audio-dir datasets/audios `
  --amt-root ../instrument_agnostic_amt
```

Preparation transcribes audio with Tsumugi and saves MIDI in
`datasets/symbolic/midi`, token caches in `datasets/symbolic/tokens`, and a
`manifest.json` containing each file's result. Use `--output-dir` to change
the destination and `--amt-python` to select the AMT Python interpreter.

After correcting a failed input or setting, rerun with `--retry-failed`.
To rebuild outdated token caches from the saved MIDI:

```powershell
uv run --extra train python -m train.symbolic_teacher.prepare_dataset --retokenize
```

If you used a custom output directory, pass the same `--output-dir` when
rebuilding caches. Checkpoints made with an incompatible vocabulary or decoder
layout require retraining.

## Train

```powershell
uv run --extra train python -m train.symbolic_teacher.train `
  --token-dir datasets/symbolic/tokens `
  --output-dir checkpoints/symbolic_teacher `
  --epochs 10
```

Training saves `last.pt` and `epoch_XXXX.pt` in the output directory.
The teacher reconstructs notes, instruments, rhythm, chords, bass, and key
from the tokenized MIDI.

`--crop-frames` sets the training window length (default: 750 frames, or
30 seconds). Use `--crop-frames 0` for whole songs. Crops retain held notes
and the musical state at the start of the window.

Use `--batch-size` to change the batch size (default: 1). Add
`--gradient-checkpointing` to reduce activation memory at the cost of additional
computation. Training automatically uses CUDA AMP when `--device cuda` (or the
default CUDA device selection) is active. `--amp-dtype auto` (the default)
selects native BF16 when supported and otherwise uses FP16; pass `float16` or
`bfloat16` to override it. CPU training remains in FP32. To train directly from
MIDI, replace `--token-dir` with `--midi-dir datasets/symbolic/midi`. Add
`--compile-encoder` to compile only the symbolic event Transformer; use
`--compile-mode` to select the `torch.compile` mode. Run with `--help` for all
options. On Windows with a non-UTF-8 locale, set `PYTHONUTF8=1` before starting
a compiled run.

## Use the teacher

Load the checkpoint into the symbolic teacher of a `TsumugiMRLPretrainingModel`
created with matching settings:

```python
import torch

checkpoint = torch.load("checkpoints/symbolic_teacher/last.pt", map_location="cpu")
model.symbolic_teacher.load_state_dict(checkpoint["state_dict"])
model.symbolic_teacher.freeze()
```

`freeze()` keeps the teacher fixed while its contrastive projection remains
trainable. Call it after loading weights, and build the optimizer from
parameters with `requires_grad=True`.

Compute teacher outputs from matching audio and MIDI clips. See the
[audio pretraining example](../../README.md#audio-pretraining).
