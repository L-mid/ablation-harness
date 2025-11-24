from ablation_harness.ema import EMA, EMAConfig


class NoOpEMA:
    def update(self, *_, **__):
        pass

    def state_dict(self, *_, **__):
        pass

    def load_state_dict(self, *_, **__):
        pass

    def copy_to(self, model): 
        pass

    def apply_to(self, model): 
        pass


def build_ema(model, cfg):
    if not getattr(cfg, "enable", False):
        return NoOpEMA()
    # support dict or dataclass
    if isinstance(cfg, dict):
        return EMA(model, EMAConfig(**cfg))
    return EMA(model, cfg)
