# Tsumugi-MRL

Tsumugi-MRL learns music representations from audio using acoustic and MIDI
teachers. The audio model produces frame features and clip embeddings for
downstream tasks.

## How it works

![Tsumugi-MRL architecture](docs/architecture.svg)

During pretraining, an audio Transformer predicts acoustic and symbolic RVQ
codes from masked audio. A contrastive objective aligns audio and MIDI clip
embeddings. Both teachers process the corresponding unmasked clip.

The acoustic teacher quantizes Mel features. The symbolic teacher encodes MIDI
events and learns discrete codes by reconstructing notes, instruments, rhythm,
chords, bass, and key.

Regenerate the figure with
`uv run python scripts/make_architecture_figure.py`.

## Extract features

Load an exported audio checkpoint and pass stereo audio sampled at 22,050 Hz
as a tensor of shape `[batch, 2, samples]`:

```python
import torch
from tsumugi_mrl import TsumugiMRLModel

model = TsumugiMRLModel.from_pretrained("checkpoints/pretraining/audio_model").eval()
with torch.no_grad():
    hidden = model(audio)  # [B, T, D], audio is [B, 2, samples]
    embedding = model.encode_embedding(audio)  # [B, projection_dim], normalized
```


For fine-tuning, call `model.train()` and enable gradients.

## Train the teachers

Run commands from the repository root. Prepare MIDI and token caches with
Tsumugi, then train the symbolic teacher:

```powershell
uv run --extra train python -m train.symbolic_teacher.prepare_dataset `
  --audio-dir datasets/audios --amt-root ../instrument_agnostic_amt

uv run --extra train python -m train.symbolic_teacher.train `
  --token-dir datasets/symbolic/tokens
```

Set `--amt-root` to your `instrument_agnostic_amt` checkout. See
[symbolic teacher training](train/symbolic_teacher/README.md) for preparation
options and [Mel-RVQ training](train/mel_rvq/README.md) to train the acoustic
teacher.

## Audio pretraining

Train with the preparation manifest and both teacher checkpoints. The default
`contrastive` mode uses all three objectives:

```powershell
uv run --extra train python -m train.train `
  --manifest datasets/symbolic/manifest.json `
  --mel-checkpoint checkpoints/mel_rvq/last.pt `
  --symbolic-checkpoint checkpoints/symbolic_teacher_weighted/last.pt `
  --output-dir checkpoints/pretraining `
  --batch-size 4 --epochs 10 --ablation contrastive
```

Use `--ablation` to select the cumulative objective set:

* `mel_rvq`: Mel-RVQ acoustic prediction only; `--symbolic-checkpoint` is not
  required.
* `symbolic_teacher`: acoustic prediction plus symbolic RVQ code prediction.
* `contrastive`: the previous two objectives plus Audio--MIDI contrastive loss.

The default loss weights are 1.0, 0.5, and 0.2, respectively. Set
`--crop-frames`, `--mask-ratio`, and `--mask-span` to change cropping and
masking. Batches require at least two pairs; the final incomplete batch is
dropped. Short clips are padded, and padding is excluded from attention,
pooling, and prediction losses.

Each epoch saves `last.pt` and `epoch_XXXX.pt`. After training, the exported
audio model is saved in `checkpoints/pretraining/audio_model` and can be loaded
with `TsumugiMRLModel.from_pretrained()`.

Resume from a completed epoch with:

```powershell
uv run --extra train python -m train.train `
  --manifest datasets/symbolic/manifest.json `
  --resume checkpoints/pretraining/last.pt `
  --output-dir checkpoints/pretraining --epochs 20
```

`--epochs` is the total epoch count. Resume restores both teachers, model
settings, optimizer, random state, and training settings (including the
ablation mode, batch size, cropping, masking, learning rate, gradient clipping,
worker count, AMP dtype, and encoder compile setting). Use the same manifest to
continue on the same dataset.

On CUDA, `--amp-dtype auto` is enabled by default. Add `--compile-encoder` to
compile only `MaskedAudioEncoder.encoder` with `torch.compile(mode="default")`.

For a new run, `--config path/to/audio_config.json` overrides audio encoder
settings: `d_model`, `n_heads`, `num_layers`, `dim_feedforward`, `dropout`, and
`gradient_checkpointing`. Other model dimensions are read from the teacher
checkpoints. Run with `--help` for all options.

The manifest contains `audio_path` and `token_path` for each pair; failed
preparation entries are skipped. Relative paths in a custom manifest are
resolved from its directory. Audio is loaded with `soundfile`, resampled, and
converted to stereo.

### Batch construction

Use `crop_pretraining_window` to crop matching audio and MIDI intervals before
computing teacher outputs. It preserves held notes and musical state at the
start of the clip and returns indices for aligning symbolic targets to audio
frames.

The example below shows batch alignment and checkpoint export. It assumes
loaded teacher weights, an audio tensor `full_audio`, and its tokenized MIDI
`sequence`.

```python
import torch
from train.config import TrainingConfig
from train.pretraining import TsumugiMRLPretrainingModel
from train.data import crop_pretraining_window
from train.symbolic import collate_symbolic_sequences

model = TsumugiMRLPretrainingModel(TrainingConfig())
# Load model.symbolic_teacher weights and mel_teacher before training.
model.symbolic_teacher.freeze()  # fixes teacher, leaves its projection trainable
model.set_mel_stats(*mel_teacher.mel_stats)
model.train()
window = crop_pretraining_window(full_audio, sequence, start_frame=250, num_frames=750, config=model.config)
batch = collate_symbolic_sequences([window.symbolic])
output = model(
    window.audio.unsqueeze(0),
    symbolic_token_ids=batch["token_ids"],
    symbolic_token_instrument_ids=batch["token_instrument_ids"],
    symbolic_token_type_ids=batch["token_type_ids"],
    symbolic_position_ids=batch["position_ids"],
    symbolic_anchor_positions=batch["anchor_positions"],
    symbolic_padding_mask=batch["padding_mask"],
    symbolic_frame_padding_mask=batch["frame_padding_mask"],
)
musical_targets = output.symbolic_codes.index_select(1, window.symbolic_frame_indices)
with torch.no_grad():
    acoustic_targets = mel_teacher.encode(window.audio.unsqueeze(0))

# After audio pretraining, save a standalone model with no teacher or SSL heads.
model.export_audio_model().save_pretrained("checkpoints/audio_model")
```

For training, apply frame masks to the audio encoder and pass the masks and
valid positions to `train.losses.PretrainingLoss`. Use unmasked clips for both
teachers and at least two pairs per batch for the contrastive loss. Move all
batch tensors and alignment indices to the model's device.

Set `TrainingConfig(gradient_checkpointing=True)` to reduce activation memory
at the cost of additional computation during backpropagation.

## Tests

```powershell
uv run --extra train python -m pytest -q tests
```

## Linear probe results

Test-set F1 from frozen audio representations. The random row uses the
random-initialized encoder; the remaining rows will be filled after evaluation.
Values are macro-F1 except for Ballroom, which uses binary F1.

<table>
<thead>
<tr>
<th rowspan="3">Pretraining</th>
<th><strong>NSynth</strong></th>
<th>NSynth</th>
<th><strong>GuitarSet</strong></th>
<th>FMA-small</th>
<th>OpenMIC</th>
<th><strong>Ballroom</strong></th>
<th><strong>GiantSteps</strong></th>
</tr>
<tr>
<th><strong>Pitch</strong></th>
<th>Instrument family</th>
<th><strong>Chord</strong></th>
<th>Genre</th>
<th>Instrument</th>
<th><strong>Beat</strong></th>
<th><strong>Key</strong></th>
</tr>
<tr>
<th>F1</th>
<th>F1</th>
<th>F1</th>
<th>F1</th>
<th>F1</th>
<th>F1</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>Random</td>
<td><strong>68.22%</strong></td>
<td>43.19%</td>
<td><strong>15.53%</strong></td>
<td>17.65%</td>
<td>25.45%</td>
<td><strong>49.33%</strong></td>
<td><strong>27.35%</strong></td>
</tr>
<tr>
<td>+ Mel RVQ</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>+ Symbolic Teacher</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>+ Contrastive</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

Bold columns are the core musical-structure probes: pitch, chord, beat, and
key.

Detailed probe commands and settings are documented in
[probes/README.md](probes/README.md).
