#!/usr/bin/env python3
"""
Dot-based quadrat map — Experiment 2 (Variable Sizes, Constant Shape and Area)
----------------------------------------------------------------
Same shape (squares) for all trials. User to define one or more square sizes.
For EACH size, compute how many quadrats are needed to match a TARGET total
sampled area (approximately, due to integer counts).

Example:
  --sizes-cm 15,20,25 --target-area-m2 0.1875
  counts ≈ round(0.1875 / 0.0225)=8, round(0.1875 / 0.04)=5, round(0.1875 / 0.0625)=3

CLI
---
--width (m)              Study area width
--height (m)             Study area height
--dot-spacing (m)        Distance between dots
--sizes-cm (list)        Comma-separated square sizes in cm (e.g., 15,20,25)
--target-area-m2 (float) Target total sampled area per size in m^2
--seed (int)             Optional RNG seed; if omitted, one is generated and printed
--out-prefix (str)       Output prefix (PNG, CSV, XLSX)

Example (one line):
`python exp2_sizes.py --width 10 --height 6 --dot-spacing 0.5 --sizes-cm 15,20,25 --target-area-m2 0.1875 --out-prefix r1`
"""

import argparse
import math
import random
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------ Data model ------------------------

@dataclass(frozen=True)
class SquareGeom:
    side: float          # side in meters
    half: float          # half side (m)

def build_square(side_m: float) -> SquareGeom:
    if side_m <= 0:
        raise ValueError("Square side must be > 0.")
    return SquareGeom(side=side_m, half=side_m / 2.0)

# Colors per size label
PALETTE = [
    "#B2DF8A",  # light green
    "#9EC9FF",  # light blue
    "#D1B3FF",  # light purple
    "#FFCC99",  # light orange
    "#FF9999",  # light red
    "#FFF9B1",  # light yellow
    "#A3E1D4",  # light teal
]

# ------------------------ Grid & masking ------------------------

def dot_grid(width: float, height: float, spacing: float) -> np.ndarray:
    if spacing <= 0:
        raise ValueError("--dot-spacing must be > 0.")
    xs = np.arange(0.0, width + 1e-9, spacing)
    ys = np.arange(0.0, height + 1e-9, spacing)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xv.ravel(), yv.ravel()])

def mask_candidates(points: np.ndarray, width: float, height: float, sq: SquareGeom) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    ok = (
        (x - sq.half >= 0.0) &
        (x + sq.half <= width) &
        (y - sq.half >= 0.0) &
        (y + sq.half <= height)
    )
    return ok

# ------------------------ Drawing ------------------------

def draw_map(width: float,
             height: float,
             points: np.ndarray,
             placements: pd.DataFrame,
             out_png: str,
             title: str,
             note_lines: List[str],
             legend_items: List[Tuple[str, str]]) -> None:

    fig, ax = plt.subplots(figsize=(10, 10 * (height / max(width, 1e-9))))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Study area boundary + dots
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, linewidth=1.2))
    ax.scatter(points[:, 0], points[:, 1], s=10, marker='o', linewidths=0, alpha=0.6)

    # Draw squares grouped by size label
    for size_label, df_group in placements.groupby("size_label"):
        color = df_group["color"].iloc[0]
        for _, r in df_group.iterrows():
            side = float(r["square_side_m"])
            half = side / 2.0
            cx, cy = float(r["center_x_m"]), float(r["center_y_m"])
            ax.add_patch(plt.Rectangle((cx - half, cy - half), side, side,
                                       facecolor=color, alpha=0.45, linewidth=1.2, edgecolor="black"))

    # Labels
    ax.set_xlabel("Meters (X)")
    ax.set_ylabel("Meters (Y)")
    ax.set_title(title)

    # Legend
    handles = [
        plt.Line2D([], [], marker='s', linestyle='None', markersize=10,
                   markerfacecolor=color, markeredgecolor="black", label=label)
        for (label, color) in legend_items
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.85, title="Quadrat Sizes")

    # Note bottom-left
    note = "\n".join(note_lines)
    ax.text(0.001, 0.001, note, transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# ------------------------ Tables ------------------------

def build_table(centers_by_size: Dict[str, np.ndarray],
                side_by_size: Dict[str, float],
                count_by_size: Dict[str, int],
                target_area_m2: float,
                color_by_size: Dict[str, str],
                meta: dict) -> pd.DataFrame:
    rows = []
    for size_label, centers in centers_by_size.items():
        side = side_by_size[size_label]
        area = side * side
        for i, (cx, cy) in enumerate(centers, start=1):
            rows.append({
                "size_label": size_label,
                "dot_index_within_size": i,
                "center_x_m": round(float(cx), 6),
                "center_y_m": round(float(cy), 6),
                "square_side_m": round(side, 6),
                "square_area_m2": round(area, 6),
                "target_total_area_m2": round(target_area_m2, 6),
                "width_m": meta["width_m"],
                "height_m": meta["height_m"],
                "dot_spacing_m": meta["dot_spacing_m"],
                "seed": meta["seed"],
                "out_prefix": meta["out_prefix"],
                "color": color_by_size[size_label],
            })
    return pd.DataFrame(rows)

def write_tables(df: pd.DataFrame, out_csv: str, out_xlsx: str) -> None:
    # Drop the color helper column before saving
    to_save = df.drop(columns=["color"])
    to_save.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        to_save.to_excel(xw, index=False, sheet_name="placements")
        xw.sheets["placements"].freeze_panes(1, 0)

# ------------------------ CLI ------------------------

def parse_sizes_cm_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Provide at least one size in cm, e.g., '15,20,25'.")
    sizes_cm = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid size '{p}'. Use numbers like 15,20,25.")
        if v <= 0:
            raise argparse.ArgumentTypeError(f"Invalid size '{p}'. Must be > 0.")
        sizes_cm.append(v)
    return sizes_cm

def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment B: same-shape (square) quadrats with VARIABLE sizes; "
                    "compute counts per size to match a TARGET sampled area (per size).")
    p.add_argument("--width", type=float, required=True, help="Study area width in meters")
    p.add_argument("--height", type=float, required=True, help="Study area height in meters")
    p.add_argument("--dot-spacing", type=float, required=True, help="Spacing between dots in meters")
    p.add_argument("--sizes-cm", type=parse_sizes_cm_list, required=True,
                   help="Comma-separated square sizes in cm, e.g., 15,20,25")
    p.add_argument("--target-area-m2", type=float, required=True,
                   help="Target TOTAL sampled area per size (in m^2), e.g., 0.1875")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--out-prefix", type=str, default="exp2_sizes", help="Output prefix (PNG, CSV, XLSX)")
    return p.parse_args()

# ------------------------ Main ------------------------

def main():
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Width/height must be > 0.")
    if args.target_area_m2 <= 0:
        raise SystemExit("--target-area-m2 must be > 0.")

    # Handle provided vs generated seed (32-bit range)
    if args.seed is not None:
        seed = int(args.seed) % (2**32)
        print(f"Seed provided: {seed}")
    else:
        seed = random.randint(0, 2**32 - 1)
        print(f"Random seed generated: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # Convert sizes to meters and compute counts to match target area
    sizes_cm = args.sizes_cm
    sizes_m = [cm / 100.0 for cm in sizes_cm]
    side_by_size: Dict[str, float] = {}
    count_by_size: Dict[str, int] = {}
    label_by_size: Dict[str, str] = {}  # e.g., "15×15 cm"

    for side_m in sizes_m:
        label = f"{int(round(side_m*100))}×{int(round(side_m*100))} cm"
        area_q = side_m * side_m
        # integer count approximating the target area
        count = max(1, int(round(args.target_area_m2 / area_q)))
        side_by_size[label] = side_m
        count_by_size[label] = count
        label_by_size[label] = label

    # Show computed counts and realized totals
    print("Computed counts and realized sampled area per size:")
    for label in label_by_size:
        s = side_by_size[label]
        c = count_by_size[label]
        total = c * s * s
        err = (total - args.target_area_m2)
        print(f"  {label}: count={c}, total={total:.6f} m^2 (target={args.target_area_m2:.6f}, Δ={err:+.6f})")

    # Build grid
    pts = dot_grid(args.width, args.height, args.dot_spacing)

    # Assign colors to sizes
    color_by_size: Dict[str, str] = {}
    for idx, label in enumerate(label_by_size):
        color_by_size[label] = PALETTE[idx % len(PALETTE)]

    # Select centers per size independently
    centers_by_size: Dict[str, np.ndarray] = {}
    for label in label_by_size:
        side = side_by_size[label]
        need = count_by_size[label]
        geom = build_square(side)
        mask = mask_candidates(pts, args.width, args.height, geom)
        candidates = pts[mask]
        if len(candidates) < need:
            raise SystemExit(
                f"[{label}] Not enough candidate dots for side={side:.3f} m.\n"
                f"Candidates: {len(candidates)}, requested: {need}.\n"
                f"Consider reducing size, increasing --dot-spacing, or enlarging the area."
            )
        idx = np.random.choice(len(candidates), size=need, replace=False)
        centers_by_size[label] = candidates[idx]

    meta = {
        "width_m": args.width,
        "height_m": args.height,
        "dot_spacing_m": args.dot_spacing,
        "seed": seed,
        "out_prefix": args.out_prefix,
    }

    # Build placement table
    df = build_table(centers_by_size, side_by_size, count_by_size,
                     target_area_m2=args.target_area_m2,
                     color_by_size=color_by_size,
                     meta=meta)

    # Title and note
    title = (f"Dot Quadrat Map — Experiment B "
             f"({args.width}m × {args.height}m; dots every {args.dot_spacing}m)")
    note_lines = [f"Seed: {seed}"]

    # Output directory: ./exp2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "exp2")
    os.makedirs(out_dir, exist_ok=True)

    # File outputs (inside exp2/)
    out_png = os.path.join(out_dir, f"{args.out_prefix}.png")
    out_csv = os.path.join(out_dir, f"{args.out_prefix}_placed.csv")
    out_xlsx = os.path.join(out_dir, f"{args.out_prefix}_placed.xlsx")

    # Legend items (label with count, color)
    legend_items = [(f"{label} ({count_by_size[label]})", color_by_size[label])
                    for label in label_by_size]

    # Draw & save
    draw_map(args.width, args.height, pts, df, out_png, title, note_lines, legend_items)
    write_tables(df, out_csv, out_xlsx)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")
    print("Done.")

if __name__ == "__main__":
    main()