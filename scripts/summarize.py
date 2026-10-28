"""Collate evaluation and planning results from several runs into markdown tables.

Example:
    python scripts/summarize.py runs/dino runs/geo runs/gauss --eval eval_eval
    python scripts/summarize.py runs/granular_* --eval eval_granular_eval

For every run directory it reads ``<eval>.json`` and, if present, the matching
no-change baseline ``<eval>_copy.json``, plus any ``plan_*.json`` files.
"""

import argparse
import json
from pathlib import Path

NAN = float("nan")


def g(d, *keys, sub="mean"):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return NAN
        d = d[k]
    if isinstance(d, dict):
        return d.get(sub, NAN)
    return d if isinstance(d, (int, float)) else NAN


OBJ_COLS = [
    ("err@1", "overall", "{r}/k1/disp_err_cm"),
    ("err@8", "overall", "{r}/k8/disp_err_cm"),
    ("moved@4", "overall", "{r}/k4/disp_err_moved_cm"),
    ("pushed@4", "locality", "{r}/k4/target_disp_err_cm"),
    ("bystander@4", "locality", "{r}/k4/spurious_motion_cm"),
    ("reappear", "persistence", "{r}/err_at_reappear_cm"),
    ("drift-hidden", "persistence", "{r}/pred_motion_during_occlusion_cm"),
]
GRAN_COLS = [
    ("change@4", "{r}/k4/change_err"),
    ("change@16", "{r}/k16/change_err"),
    ("dug", "{r}/changed_err"),
    ("spurious", "{r}/spurious"),
    ("kept", "{r}/kept_ratio"),
]


def object_rows(name, r, readout):
    out = []
    for view in ("train_cams", "heldout_cam"):
        if view not in r:
            continue
        v = r[view]
        cells = [f"{g(v.get(sec, {}), key.format(r=readout)):.2f}" for _, sec, key in OBJ_COLS]
        out.append(f"| {name} | {view} | " + " | ".join(cells) + " |")
    return out


def granular_rows(name, r, readout):
    out = []
    for view in ("train_cams", "heldout_cam"):
        if view not in r:
            continue
        gr = r[view].get("granular", {})
        cells = [f"{g(gr, key.format(r=readout)):.2f}" for _, key in GRAN_COLS]
        out.append(f"| {name} | {view} | " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--eval", default="eval_eval", help="evaluation file stem inside each run directory")
    args = ap.parse_args()
    obj, gran, plans = [], [], []
    for run in map(Path, args.runs):
        files = sorted(run.glob(f"{args.eval}*.json"))
        for f in files:
            r = json.load(open(f))
            tag = run.name + (" [no-change]" if f.stem.endswith("_copy") else "")
            readouts = ["probe"] + (["explicit"] if r.get("rep") == "gauss" and not f.stem.endswith("_copy") else [])
            for rd in readouts:
                name = tag + ("" if rd == "probe" else " [explicit]")
                if "granular" in r.get("train_cams", {}):
                    gran += granular_rows(name, r, rd)
                else:
                    obj += object_rows(name, r, rd)
        for pf in sorted(run.glob("plan_*.json")):
            pr = json.load(open(pf))
            plans.append(f"| {pr['name']} | {pr['success_rate']:.2f} | {pr['mean_final_dist_cm']:.1f} | {pr['n_tasks']} |")
    for pf in sorted(Path("runs").glob("plan_*.json")):
        pr = json.load(open(pf))
        plans.append(f"| {pr['name']} | {pr['success_rate']:.2f} | {pr['mean_final_dist_cm']:.1f} | {pr['n_tasks']} |")
    if obj:
        print("Object metrics, cm, lower is better.\n")
        print("| run | view | " + " | ".join(c for c, _, _ in OBJ_COLS) + " |")
        print("|" + "---|" * (len(OBJ_COLS) + 2))
        print("\n".join(obj))
    if gran:
        print("\nGranular metrics, grains per cell (kept: fraction of the dug change still predicted at the end, 1 is best).\n")
        print("| run | view | " + " | ".join(c for c, _ in GRAN_COLS) + " |")
        print("|" + "---|" * (len(GRAN_COLS) + 2))
        print("\n".join(gran))
    if plans:
        print("\n| planner | success | final dist (cm) | tasks |\n|---|---|---|---|")
        print("\n".join(dict.fromkeys(plans)))


if __name__ == "__main__":
    main()
