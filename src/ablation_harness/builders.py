from ablation_harness.ema import EMA


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
    if not getattr(cfg, "ema_enabled", False):
        print("[debug builders] ema noop path")
        return NoOpEMA()
    # support dict or dataclass
    print("[debug builders] ema path selected")
    return EMA(model, cfg)
