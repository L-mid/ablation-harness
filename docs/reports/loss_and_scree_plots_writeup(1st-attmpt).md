## Loss & Scree plots writeup wk1


# Problem & Hypothesis:
Recently I've been coding lots of ML infra frameworks and ideas, but with no consolidation or practice in shipping.
Becase of this: no ablations, stats, plots, etc (reaserch framework).

My idea is that by shipping smaller projects, it teaches the whole story and what you need to know by practice, not just the half of it.
So I'm going to build and ship a toy ablator with some plots, here's some from the first week.


# Setup:

data slice: CIFAR10
    subset: 256,
    batch_size 128.
    32x32, simple ToTensor() transforms.

Model: TinyCNN.
    Designed for <10k params (has 7738), <10 sec CPU forward time.


My baseline yaml to compare to graphs/results: (highlights):

    metric: "val/acc"       <- can be used by plot_loss.py for dir to metric

    baseline:
    model: "tinycnn"
    epochs: 1
    batch_size: 256
    dataset: "cifar10"      <- model configuations + training settings
    ...

    spectral_diag:          <- for plot_scree.py, track each layer's top singular values (set high for the plotter)
        topk: 5
        ...

    seeds: [0, 1]           <- sweep seeds

compute: all done & tested on CPU, CUDA untested.

run length: ~90 secs + run plt commands (few secs each)


command for training + recording ablations:
    python -m ablation_harness.ablate --config experiments/baseline.yaml --out_dir runs/writeup_baseline --seed 11

plotting commands:
    plot_loss:
        python -m ablation_harness.plot_loss runs/logs/id_writeup_baseline_1/loss.jsonl --metrics train/loss --out runs/writeup_baseline/loss

    plot_scree:
        python -m ablation_harness.plot_scree runs/logs/id_writeup_baseline_1/spectral_final.json --layer classifier --out runs/writeup_baseline/figs/scree_classifier --logy --normalize



# Method
- added ablate.py:
    - parses CI params,
    - verifies yaml config schema,
    - records data from trainer.py,
    - dumps to data in jsonl + saves in specified dir.

Recorded data includes:
    val/acc, val/loss, params, ckpt, dataset & model used,
    spect_stats (nesting more complex ablation data),
    + run_id and seed used in that run.

- added two plotters: plot_loss.py & plot_scree.py:
    - both are specified to a dir (json or jsonl),
    - plot_loss graphs loss curve over pts,
    - plot_scree graphs scree, in a layer.


# Results
The loss plot:
![Loss plot](ablation-harness/runs/writeup_baseline/loss/loss.png)
- shows curve over many points
- --ylim: default zooms in on tight ys, show even fractional/dead curves, can be set
- examples of various utilies: labeling, ema smoothing, collects from jsonl + csv, 5+ params

The scree plot:
![Scree plot](ablation-harness/runs/writeup_baseline/figs/scree_classifier.png)
- view of all k in a signle layer (classifier as example).
- can be set --ylog, makes decay readable
- --normalize param to divide by σ₁ so curves are in [0,1]


# Interpretation
3 takeaways:

Helps with stats collection, great first start, layers + data now actually readable, baselines recorded.

Great potential if/when hooked with a logger (wandb or tensorboard).

Hurts in complexity, some plotters could be simpler, and use same infra to work one day. Some feats in plotters not needed yet (plot_loss.py ema smoothing), just add complexity before needed.



# Next steps:

Todo:
    Test the scree plotter properly: make pytests.
    Set their outputs additonally to logging hooks (wandb, tensorboard).
    Refactor the loss plotter and scree under a unified merged plotting API if used a lot together/lots of future plots as plugins.
    Refactor spectral.py stats collection to occur under unified dir like plot_loss + others (plot_scree itself can collect from wherever directed)
