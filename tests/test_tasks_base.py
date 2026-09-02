import numpy as np
import torch

from bdhx.tasks.base import Episode, EpisodicTask, pad_and_batch
from bdhx.tasks.vocab import (
    ANSWER,
    BOS,
    EOS,
    PAD,
    SYMBOL_OFFSET,
    VOCAB_SIZE,
    apply_permutation,
    draw_symbols,
    reserved_token,
    symbol_permutation,
)


class DummyTask(EpisodicTask):
    name = "dummy"

    def sample(self, rng, difficulty):
        raise NotImplementedError

    def train_difficulties(self):
        return [{"n": 1}]

    def eval_difficulties(self):
        return {"interp": [{"n": 1}], "mild": [{"n": 2}], "strong": [{"n": 4}]}

    def score(self, prediction, episode):
        return self.base_score(prediction, episode)


def make_episode(n_demos=3, split="train"):
    return Episode(
        demonstrations=[
            (torch.tensor([100 + i]), torch.tensor([200 + i, 201 + i])) for i in range(n_demos)
        ],
        query=torch.tensor([101]),
        target=torch.tensor([201, 202]),
        difficulty={"n_bindings": n_demos},
        split=split,
        episode_id=n_demos,
    )


def test_serialize_parse_roundtrip():
    task = DummyTask()
    ep = make_episode()
    ser = task.serialize(ep)
    assert ser[0].item() == BOS and ser[-1].item() == EOS
    back = task.parse_serialized(
        ser, difficulty=ep.difficulty, split=ep.split, episode_id=ep.episode_id
    )
    assert len(back.demonstrations) == len(ep.demonstrations)
    for (a, b), (c, d) in zip(back.demonstrations, ep.demonstrations):
        assert torch.equal(a, c) and torch.equal(b, d)
    assert torch.equal(back.query, ep.query)
    assert torch.equal(back.target, ep.target)
    assert back.difficulty == ep.difficulty


def test_pad_and_batch_shapes():
    task = DummyTask()
    eps = [make_episode(2, "train"), make_episode(5, "strong")]
    batch = pad_and_batch(eps, task)
    assert len(batch) == 2
    assert batch.query.shape == (2, 1)
    assert batch.target.shape == (2, 2)
    assert batch.demonstrations.shape[0] == 2
    assert batch.demonstrations.shape[1] == len([t for t in task.serialize(eps[1]).tolist()]) - (
        1 + eps[1].query.numel() + 1 + eps[1].target.numel() + 1
    )
    assert batch.demonstrations_mask[0].sum() < batch.demonstrations_mask[1].sum()
    assert batch.demonstrations[0][batch.demonstrations_mask[0].logical_not()].eq(PAD).all()
    assert batch.splits == ["train", "strong"]
    assert batch.difficulties[1] == {"n_bindings": 5}
    for i, ep in enumerate(eps):
        start = int(batch.answer_start[i])
        assert batch.serialized[i, start] == ep.target[0]
        assert batch.serialized[i, start - 1] == ANSWER


def test_base_score():
    task = DummyTask()
    ep = make_episode()
    assert task.score(torch.tensor([201, 202]), ep) == {"exact_match": 1.0, "token_acc": 1.0}
    s = task.score(torch.tensor([201, 999]), ep)
    assert s["exact_match"] == 0.0 and s["token_acc"] == 0.5


def test_vocab_helpers():
    rng = np.random.default_rng(0)
    syms = draw_symbols(rng, 8)
    assert len(set(syms.tolist())) == 8
    assert (syms >= SYMBOL_OFFSET).all() and (syms < VOCAB_SIZE).all()
    perm = symbol_permutation(rng)
    toks = torch.tensor([BOS, int(syms[0]), ANSWER, int(syms[1])])
    mapped = apply_permutation(toks, perm)
    assert mapped[0] == BOS and mapped[2] == ANSWER
    assert mapped[1] != toks[1] or mapped[3] != toks[3]
    assert reserved_token(0) == 8
