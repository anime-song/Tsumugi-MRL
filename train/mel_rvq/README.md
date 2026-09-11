# Mel-RVQ training

Train an audio tokenizer to produce acoustic targets for Tsumugi-MRL pretraining.

## Train

Place WAV, FLAC, or OGG files in an audio directory and run from the repository
root:

```powershell
uv run --extra train python -m train.mel_rvq.train `
  --audio-dir datasets/audios `
  --output-dir checkpoints/mel_rvq `
  --epochs 20
```

The loader resamples audio and converts it to stereo automatically.
`--clip-seconds` sets the training clip length (default: 4 seconds), and
`--batch-size` sets the batch size (default: 8).

Training can also read the Mel cache built by
`scripts/prepare_pretraining_cache.py`, which skips audio decoding and the
STFT:

```powershell
uv run --extra train python -m train.mel_rvq.train `
  --mel-manifest datasets/symbolic/pretraining_cache/manifest.json `
  --output-dir checkpoints/mel_rvq --epochs 20
```

Pass exactly one of `--audio-dir` or `--mel-manifest`. Cached features are
already normalized, so the statistics cannot be estimated from them: copy the
`mel_stats.json` used to build the cache into the output directory, or point
`--stats-path` at it.

Training saves `last.pt` and `epoch_XXXX.pt` in the output directory.
Mel normalization statistics are estimated on the first run and saved in
`mel_stats.json` and each checkpoint. Use `--stats-clips N` to limit the number
of clips used for estimation.

The tokenizer quantizes the folded Mel features directly. It defaults to 8
codebooks and 1,024 entries per codebook, and each quantizer stage projects the
residual down to a 16-dimensional codebook space before the lookup, then back
up again. Change these with `--codebooks`, `--codebook-size`, and
`--codebook-dim`. An entry that goes unused for `--stale-tolerance` steps is
revived from the current batch, which keeps codebooks from dying. Run with
`--help` for all options.

Each log line reports the codebook perplexity, the effective number of entries
each stage uses. It is logged per stage and as an average, and a stage that
collapses shows up there before the loss moves.

For Weights & Biases logging, run `uv run --extra train wandb login` and add
`--wandb` to the training command. Set `--wandb-project` and `--wandb-name` to
choose the project and run name.

## Published checkpoint

The released Mel-RVQ tokenizer is available from the
[Hugging Face Hub](https://huggingface.co/anime-song/tsumugi-mrl-mel-rvq):

```python
from train.mel_rvq.model import MelRVQTokenizer

teacher = MelRVQTokenizer.from_pretrained("anime-song/tsumugi-mrl-mel-rvq")
```

The Hub checkpoint quantizes the folded Mel features directly with eight
residual stages of 1,024 entries, each projecting through a 16-dimensional
codebook space, and includes the Mel normalization statistics used during
training. It produces fixed acoustic targets for Tsumugi-MRL pretraining; it
does not synthesize audio.

The published weights follow the quantizer layout described above, so load them
with a matching checkout of this repository. Checkpoints from before that
change are not compatible.

## Use the tokenizer

Pass unmasked stereo audio at 22,050 Hz as `[batch, 2, samples]`:

```python
import torch
from train.mel_rvq.model import MelRVQTokenizer

teacher = MelRVQTokenizer.from_checkpoint("checkpoints/mel_rvq/last.pt")
with torch.no_grad():
    acoustic_targets = teacher.encode(unmasked_audio)
# acoustic_targets: [batch, time, codebooks]
```

Use `from_checkpoint` for local training `.pt` files and `from_pretrained` for
the Hub release.

Before audio pretraining, copy the teacher's normalization statistics to the
audio model:

```python
model.set_mel_stats(*teacher.mel_stats)
```

Use `acoustic_targets` with `PretrainingLoss`. See the
[audio pretraining example](../../README.md#audio-pretraining) for paired clips
and target alignment.
