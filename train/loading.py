"""Resolve teacher weights from the Hub or from a local training checkpoint."""

from pathlib import Path

# The released teachers. Entry points fall back to these so a fresh checkout can
# pretrain without first training or downloading anything by hand.
MEL_RVQ_SOURCE = "anime-song/tsumugi-mrl-mel-rvq"
SYMBOLIC_TEACHER_SOURCE = "anime-song/tsumugi-mrl-symbolic-teacher"


def load_teacher(model_class, source=None, default=None):
    """Load a teacher from a ``.pt`` checkpoint, an export directory, or the Hub.

    Training writes ``.pt`` files that carry a config and a state dict, while
    ``save_pretrained`` and the Hub use safetensors beside a ``config.json``.
    The two need different loaders, so pick by what ``source`` names: a path
    ending in ``.pt`` is a training checkpoint, and anything else is a Hub
    repository id or a directory holding an export.

    ``source`` of ``None`` falls back to ``default``, which is how the entry
    points reach the published weights without naming a local path.
    """

    chosen = source if source is not None else default
    if chosen is None:
        raise ValueError(f"No source given for {model_class.__name__}.")
    path = Path(chosen)
    if path.suffix == ".pt":
        if not path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
        return model_class.from_checkpoint(path)
    return model_class.from_pretrained(str(chosen))
