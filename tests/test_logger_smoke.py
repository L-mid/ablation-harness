"""Add test for why wandb isn't working + the dir path/not listen to dataclass problem."""

from ablation_harness.logger import MultiLogger, NullLogger


def test_null_logger_no_crash():
    lg = NullLogger()
    lg.on_run_start({"a": 1})
    lg.on_epoch_start(0)
    lg.log_metrics({"x": 1.0}, step=0)
    lg.log_text("msg", "hello", step=0)
    lg.log_artifact(__file__, name="self")
    lg.on_epoch_end(0)
    lg.on_run_end()


def test_multi_logger_empty_ok():
    lg = MultiLogger([NullLogger(), NullLogger()])
    lg.log_metrics({"y": 2.0}, step=1)
