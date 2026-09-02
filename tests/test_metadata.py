import json

from bdhx.metadata import METADATA_FIELDS, collect_metadata, update_metadata


def test_all_fields_present(tiny_cfg, results_dir):
    cfg = tiny_cfg()
    meta = collect_metadata(cfg, results_dir)
    on_disk = json.loads((results_dir / "metadata.json").read_text())
    for field in METADATA_FIELDS:
        assert field in meta
        assert field in on_disk
    assert meta["torch_version"]
    assert meta["gpu_name"] is None or isinstance(meta["gpu_name"], str)
    assert meta["config"]["experiment"]["name"] == "tiny_smoke"
    assert (results_dir / "pip_freeze.txt").exists()


def test_update_metadata(tiny_cfg, results_dir):
    collect_metadata(tiny_cfg(), results_dir)
    meta = update_metadata(results_dir, steps_completed=40, param_total=1234)
    assert meta["steps_completed"] == 40
    reloaded = json.loads((results_dir / "metadata.json").read_text())
    assert reloaded["param_total"] == 1234
