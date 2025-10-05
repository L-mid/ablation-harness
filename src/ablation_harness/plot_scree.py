"""
Useage:
    python -m ablation_harness.plot_scree runs/ckpts/<filenamewithrunid>/spectral_final.json --layer classifier --out runs/scree_classifier --logy

    --layer name like classifier, features.0, etc.
    --out is a path prefix; it writes <out>.png.
    --logy optional: log scale on y to make decay readable.
    --normalize optional: divide by σ₁ so curves are in [0,1].

This plotter only accepts json/dict input.

current command:
    python -m ablation_harness.plot_scree runs/ckpts/tinycnn=cifar10-s101-1759653433/spectral_final.json --layer classifier --out runs/scree_classifier --logy


No pytests for it.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt


def load_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support either {"stats": {...}} or a bare dict of layer->rec
    if isinstance(data, dict) and "stats" in data:
        stats = data["stats"]
    else:
        stats = data
    if not isinstance(stats, dict):
        raise ValueError("Unrecognized JSON format; expected a dict of layer->record.")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot SVD scree (singular values) for a layer.")
    p.add_argument("path", help="Path to spectral_final.json (or any JSON with {layer: {sv_topk: [...]}})")
    p.add_argument("--layer", required=False, help="Layer name to plot, e.g. 'classifier' or 'features.0'")
    p.add_argument("--out", required=False, help="Output path prefix (writes <out>.png). Defaults next to JSON.")
    p.add_argument("--logy", action="store_true", help="Use log scale for y-axis.")
    p.add_argument("--normalize", action="store_true", help="Normalize by σ₁ (divide all σ by the max).")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    stats = load_stats(args.path)
    layers = list(stats.keys())
    if not layers:
        raise SystemExit("No layers found in stats JSON.")

    layer = args.layer or layers[0]
    if layer not in stats:
        print(f"[ERROR] Layer '{layer}' not found.\nAvailable layers:\n  - " + "\n  - ".join(layers))
        raise SystemExit(2)

    rec = stats[layer]
    if "sv_topk" not in rec or not rec["sv_topk"]:  # we saved the top-k; for final we tried to save many
        raise SystemExit(f"Layer '{layer}' has no 'sv_topk' array.")
    s = rec["sv_topk"]
    # Ensure descending for sanity:
    if any(s[i] < s[i + 1] for i in range(len(s) - 1)):
        print("[WARN] singular values are not monotonically decreasing — check your collector.")

    y = list(s)
    if args.normalize and y:
        m = max(y)
        if m > 0:
            y = [v / m for v in y]

    # Out path
    if args.out:
        out_png = f"{args.out}.png"
    else:
        base = os.path.splitext(args.path)[0]  # out_dir defaults to next to loaded jsonl
        out_png = f"{base}_scree_{layer.replace('.', '_')}.png"
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(y) + 1), y, marker="o")
    plt.xlabel("index i")
    plt.ylabel("singular value σ_i" + (" (normalized)" if args.normalize else ""))
    plt.title(f"Scree plot — {layer}")
    if args.logy:
        plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi)
    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
