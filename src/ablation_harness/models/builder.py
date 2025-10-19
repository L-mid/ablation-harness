import torch.nn as nn

from .tinycnn import TinyCNN

# ----------------------
# Models
# ----------------------


# MLP (default)
class MLP(nn.Module):
    def __init__(self, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def build_model(cfg) -> nn.Module:  # chooses MLP vs TinyCNN

    # --- Data ---
    if cfg.dataset == "moons":
        _, num_classes = (2,), 2
    elif cfg.dataset == "cifar10":
        _, num_classes = (3, 32, 32), 10
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # --- Model ---
    if cfg.model_name == "mlp":
        model = MLP(hidden=cfg.hidden, dropout=cfg.dropout)
    else:
        model = TinyCNN(num_classes=num_classes, dropout=cfg.dropout)

    return model
