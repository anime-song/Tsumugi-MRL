import pytest
import torch

from train.symbolic_teacher.train import (
    _build_symbolic_loss_weights,
    _f1_from_counts,
    _macro_f1_from_counts,
    _perplexity_from_counts,
    build_parser,
)


def test_symbolic_teacher_parser_accepts_wandb_options():
    args = build_parser().parse_args(
        [
            "--wandb",
            "--wandb-project",
            "project",
            "--wandb-name",
            "run",
            "--wandb-entity",
            "entity",
            "--log-interval",
            "5",
            "--val-ratio",
            "0.2",
        ]
    )

    assert args.wandb is True
    assert args.wandb_project == "project"
    assert args.wandb_name == "run"
    assert args.wandb_entity == "entity"
    assert args.log_interval == 5
    assert args.val_ratio == 0.2
    assert args.beat_pos_weight == 5.0
    assert args.downbeat_pos_weight == 20.0
    assert args.max_note_pos_weight == 50.0
    assert args.balanced_softmax_tau == 0.3


def test_validation_metric_helpers_compute_f1_and_rvq_perplexity():
    assert _f1_from_counts(2, 1, 1) == 2 / 3
    counts = {
        "true_positive": torch.tensor([2, 1]),
        "predicted": torch.tensor([3, 1]),
        "support": torch.tensor([3, 2]),
    }
    assert _macro_f1_from_counts(counts) == (2 / 3 + 2 / 3) / 2
    code_counts = torch.tensor([[2.0, 2.0], [4.0, 0.0]])
    assert _perplexity_from_counts(code_counts) == pytest.approx(1.5)


def test_symbolic_loss_weights_use_sparse_counts_and_amt_defaults():
    statistics = {
        "binary": {
            "note_onset": {
                "positive": torch.tensor(1.0),
                "negative": torch.tensor(99.0),
            },
            "note_offset": {
                "positive": torch.tensor(2.0),
                "negative": torch.tensor(98.0),
            },
        },
        "categorical": {
            "chord": torch.tensor([10.0, 2.0]),
            "meter": torch.tensor([20.0, 1.0]),
        },
    }
    binary, categorical = _build_symbolic_loss_weights(
        statistics,
        beat_pos_weight=5.0,
        downbeat_pos_weight=20.0,
        max_note_pos_weight=100.0,
    )

    assert binary["note_onset"].item() == 99.0
    assert binary["note_offset"].item() == 49.0
    assert binary["beat"].item() == 5.0
    assert binary["downbeat"].item() == 20.0
    assert torch.equal(categorical["chord"], statistics["categorical"]["chord"])


def test_length_bucket_sampler_groups_similar_lengths():
    from train.symbolic_teacher.train import LengthBucketSampler

    lengths = [100, 900, 110, 880, 105, 890]
    sampler = LengthBucketSampler(lengths, batch_size=3, seed=0)

    batches = list(sampler)
    assert len(sampler) == len(batches) == 2
    assert sorted(index for batch in batches for index in batch) == list(range(6))
    # Short items land together and long ones together, whatever the order.
    grouped = sorted(sorted(lengths[index] for index in batch) for batch in batches)
    assert grouped == [[100, 105, 110], [880, 890, 900]]

    # A different epoch may reorder the batches but keeps the grouping.
    sampler.set_epoch(3)
    regrouped = sorted(sorted(lengths[index] for index in batch) for batch in list(sampler))
    assert regrouped == grouped


def test_length_bucket_sampler_handles_a_partial_batch():
    from train.symbolic_teacher.train import LengthBucketSampler

    sampler = LengthBucketSampler([10, 20, 30, 40, 50], batch_size=2, seed=1)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 3
    assert sorted(len(batch) for batch in batches) == [1, 2, 2]

    dropped = LengthBucketSampler([10, 20, 30, 40, 50], batch_size=2, seed=1, drop_last=True)
    assert len(list(dropped)) == len(dropped) == 2
