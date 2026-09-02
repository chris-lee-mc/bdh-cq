import pytest
import yaml
from pydantic import ValidationError

from bdhx.config import (
    Config,
    apply_dotted_overrides,
    canonical_yaml,
    config_hash,
    load_config,
)

BASE = {
    "experiment": {"name": "x", "stage": "dev", "tags": []},
    "model": {"name": "bdh_cq", "params_target": 10_000_000},
    "task": {"name": "compose", "seed": 7},
    "training": {"steps": 10, "batch_size": 4},
}


def test_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        Config.model_validate({**BASE, "bogus": 1})
    with pytest.raises(ValidationError):
        Config.model_validate({**BASE, "training": {"steps": 1, "batch_size": 1, "lrx": 3}})


def test_hash_stable_across_key_order():
    a = Config.model_validate(BASE)
    reordered = {k: BASE[k] for k in reversed(list(BASE))}
    reordered["model"] = {"params_target": 10_000_000, "name": "bdh_cq"}
    b = Config.model_validate(reordered)
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 12
    assert canonical_yaml(a) == canonical_yaml(b)


def test_underscore_numbers_and_string_ints():
    c = Config.model_validate({**BASE, "model": {"name": "m", "params_target": "10_000_000"}})
    assert c.model.params_target == 10_000_000
    assert yaml.safe_load("a: 10_000_000")["a"] == 10_000_000


def test_extends(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(BASE))
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump({"extends": str(base), "training": {"steps": 99}, "model": {"depth": 4}})
    )
    cfg = load_config(child)
    assert cfg.training.steps == 99
    assert cfg.training.batch_size == 4  # merged from base
    assert cfg.model.depth == 4
    assert cfg.model.name == "bdh_cq"


def test_dotted_overrides():
    d = apply_dotted_overrides({"a": {"b": 1}}, {"a.b": 2, "a.c.d": "3"})
    assert d == {"a": {"b": 2, "c": {"d": 3}}}
    cfg = load_config(
        "configs/base/tiny_smoke.yaml", {"training.lr": 1e-3, "model.recurrence.kind": "residual"}
    )
    assert cfg.training.lr == 1e-3
    assert cfg.model.recurrence.kind == "residual"


def test_tiny_smoke_loads(tiny_cfg):
    cfg = tiny_cfg()
    assert cfg.experiment.name == "tiny_smoke"
    assert cfg.model.params_target == 200_000
    assert cfg.training.steps == 40
    assert cfg.compute.deterministic is True
    assert cfg.training.loss == "final_answer"
    assert cfg.model.ttt is None
    assert cfg.evaluation.record_adaptation_cost is False


def test_config_is_frozen(tiny_cfg):
    cfg = tiny_cfg()
    with pytest.raises(ValidationError):
        cfg.training.steps = 1
