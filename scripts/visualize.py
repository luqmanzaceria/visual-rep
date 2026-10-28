"""Qualitative rollout figures.

Top panel: top-down view of the table with ground-truth object trajectories (solid)
and the model's read-out trajectories (dashed). Bottom panel: the camera frames of
the episode and, for Gaussians, the rendered predicted scene under each frame.

Example:
    python scripts/visualize.py runs/gauss --data data/eval --kind sweep --index 0 --horizon 8 --out figs/gauss_sweep.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vrep.evaluation.metrics import rollout_positions
from vrep.training import load_experiment
from vrep.training.rollout import rollout_scene

OBJ_STYLE = [  # colour follows the object's identity; marker shape is the secondary encoding
    dict(color="#c8323c", marker="s", label="red cube"),
    dict(color="#2e8b3a", marker="o", label="green sphere"),
    dict(color="#3a5bd0", marker="^", label="blue cylinder"),
]


def topdown(ax, exp, res, b, ep, t0, horizon):
    th = exp.store.scene.table_half
    ax.add_patch(plt.Rectangle((-th, -th), 2 * th, 2 * th, fill=False, lw=1, color="#888888"))
    pp = ep.pusher_pos[t0 : t0 + horizon + 1, :2]
    ax.plot(pp[:, 0], pp[:, 1], color="#444444", lw=1.5, label="pusher (true)")
    ax.scatter(pp[0, 0], pp[0, 1], color="#444444", s=60, zorder=3)
    for j, st in enumerate(OBJ_STYLE):
        gt = res.gt_xy[b, :, j]
        pred = np.concatenate([res.gt_xy[b, :1, j], res.gt_xy[b, 0, j] + (res.pred_xy[b, :, j] - res.ref_xy[b, j])], 0)
        ax.plot(gt[:, 0], gt[:, 1], color=st["color"], lw=2, marker=st["marker"], ms=5, label=f"{st['label']} (true)")
        ax.plot(pred[:, 0], pred[:, 1], color=st["color"], lw=2, ls="--", marker=st["marker"], ms=5, mfc="white", label=f"{st['label']} (model)")
        if res.explicit_xy is not None:
            ex = res.explicit_xy[b, :, j]
            ex = res.gt_xy[b, 0, j] + (ex - ex[:1])
            ax.plot(ex[:, 0], ex[:, 1], color=st["color"], lw=1, ls=":", alpha=0.8)
    ax.set_xlim(-th - 0.02, th + 0.02)
    ax.set_ylim(-th - 0.02, th + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--data", default=None)
    p.add_argument("--kind", default="push")
    p.add_argument("--index", type=int, default=0, help="which episode of that kind")
    p.add_argument("--cam", type=int, default=None, help="default: the sweep camera for sweeps, else 0")
    p.add_argument("--t0", type=int, default=None)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    exp = load_experiment(args.run, device=args.device, data=args.data)
    store = exp.store
    pool = list(range(len(store))) if args.data else exp.val_eps
    eps = [e for e in pool if store[e].kind == args.kind]
    e = eps[args.index]
    ep = store[e]
    cam = args.cam if args.cam is not None else (ep.cam if ep.cam >= 0 else 0)
    if args.t0 is None:
        t0 = 0
        if ep.kind == "sweep":
            from vrep.evaluation.metrics import find_occlusion

            occ = find_occlusion(ep, cam, ep.target)
            if occ is not None:
                t0 = occ[0]
    else:
        t0 = args.t0
    horizon = min(args.horizon, store.T - t0)
    items = [(e, t0, cam)]
    res = rollout_positions(exp, items, horizon)

    n = horizon + 1
    rows = 3 if (exp.is_gauss or exp.decoder is not None) else 2
    fig = plt.figure(figsize=(max(8, 1.3 * n), 3.2 * rows))
    gs = fig.add_gridspec(rows, n, height_ratios=[2.2] + [1] * (rows - 1))
    ax = fig.add_subplot(gs[0, : max(1, n // 2)])
    topdown(ax, exp, res, 0, ep, t0, horizon)
    err = res.disp_err[0]
    ax.set_title(f"{exp.rep.name} | {ep.kind} ep {e} cam {cam} t0={t0} | disp err @{horizon}: " + ", ".join(f"{err[-1, j] * 100:.1f}cm" for j in range(3)), fontsize=9)
    for k in range(n):
        a = fig.add_subplot(gs[1, k])
        a.imshow(ep.rgb[t0 + k, cam])
        a.set_title(f"t={t0 + k}", fontsize=8)
        a.axis("off")
    if exp.is_gauss:
        rng = np.random.default_rng(0)
        scene = exp.scene_items(items, rng)
        actions = torch.from_numpy(ep.actions[t0 : t0 + horizon][None]).float().to(exp.device)
        scenes, _ = rollout_scene(exp.predictor, exp.rep, scene, actions)
        R, t = exp.cam_RT([cam])
        bg = torch.tensor([0.33, 0.34, 0.38], device=exp.device)
        with torch.no_grad():
            for k, sc in enumerate([scene] + scenes):
                img = sc.render(exp.K, R, t, 64, 64, render_scale=0.5, background=bg)["rgb"][0].clamp(0, 1).cpu().numpy()
                a = fig.add_subplot(gs[2, k])
                a.imshow(img)
                a.set_title("init" if k == 0 else f"pred +{k}", fontsize=8)
                a.axis("off")
    elif exp.decoder is not None:
        from vrep.training.rollout import rollout_tokens

        tokens = exp.encode_items(items)
        actions = torch.from_numpy(ep.actions[t0 : t0 + horizon][None]).float().to(exp.device)
        preds, _ = rollout_tokens(exp.predictor, exp.rep, tokens, actions)
        with torch.no_grad():
            for k, tb in enumerate([tokens] + preds):
                img = exp.decoder(tb.to("cpu"))[0].clamp(0, 1).numpy()
                a = fig.add_subplot(gs[2, k])
                a.imshow(img)
                a.set_title("decoded" if k == 0 else f"pred +{k}", fontsize=8)
                a.axis("off")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.run) / f"viz_{ep.kind}_{e}_cam{cam}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
