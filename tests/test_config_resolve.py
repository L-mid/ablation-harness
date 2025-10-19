from ablation_harness.config_resolve import resolve_config, resolve_spec, to_runtime


def test_unknown_keys_raise():
    """Tests if config_resolve.py catches and fails unknown fields."""
    try:
        resolve_spec({"not_a_field": 123})
    except KeyError:
        assert True
    else:
        assert False


def test_runtime_flattening():
    """Test if to_runtime flattens the dict to pure rt cfg. (tests overrides/presets merge)"""
    spec = resolve_spec({"study_name": "s", "model": {"name": "tinycnn", "dropout": 0.3}, "data": {"dataset": "cifar10", "batch_size": 128}})
    rt = to_runtime(spec)
    assert rt.model_name == "tinycnn"
    assert rt.dropout == 0.3
    assert rt.dataset == "cifar10"
    assert rt.batch_size == 128


def test_resolve_nested_tinycnn():
    """Test to ensure config successfully overrides the dataclass defaults."""
    cfg = {
        "data": {"dataset": "cifar10"},
        "model": {"name": "tinycnn", "hidden": 64, "dropout": 0.0},
        "optim": {"optimizer": "adam", "lr": 1e-3},
    }
    rt, _ = resolve_config(cfg)
    assert rt.model_name == "tinycnn"
    assert rt.dataset == "cifar10"
