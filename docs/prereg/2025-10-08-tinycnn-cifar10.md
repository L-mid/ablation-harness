# Prereg - TinyCNN on CIFAR-subset: optimizer x lr x EMA
**Primary metric:** val/acc (maximize)
**Secondary:** val/loss; wall-time per epoch
**Dataset:** CIFAR-10 subset=256; fixed class split & transforms
**Model:** TinyCNN (7-9k params); fixed arch & dropout=0
**Training:** epochs=25, batch=128, wd ∈ {0, 5e-4}, lr as below, seed=0
**Grid (6):**
1) Adam, lr=1e-3, wd=0, EMA=off
2) Adam, lr=3e-4, wd=0, EMA=off
3) Adam, lr=1e-3, wd=0, EMA=on(β=0.999)
4) SGD(m=0.9), lr=1e-2, wd=5e-4, EMA=off
5) SGD(m=0.9), lr=3e-3, wd=5e-4, EMA=off
6) SGD(m=0.9), lr=3e-3, wd=5e-4, EMA=on(β=0.999)

**Hypotheses:**
H1: Adam@1e-3 > Adam@3e-4 on this tiny regime.
H2: EMA helps only for SGD at small lr.

**Analysis plan:**
- Aggregate `val/acc` and `val/loss` at final epoch.
- Plot loss curves.
- Bar chart: `val/acc` per config
- (Optional) Next week: 3 seeds for top-2 configs → seed variance plot.
**Deviations:** None planned.

**Commit/tag:** <link to commit>.
