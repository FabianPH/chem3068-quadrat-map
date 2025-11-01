#!/usr/bin/env python3
"""
Dot-based quadrat map — Experiment 3 (Phase 1: Randomization only)
------------------------------------------------------------------
Place SQUARE quadrats at randomized dot-lattice points.

CLI
---
--width (m)             Study area width
--height (m)            Study area height
--dot-spacing (m)       Distance between dots
--shape-area (cm^2)     Square quadrat area in cm^2 (for one quadrat)
--n-quadrats            Number of quadrats to place
--seed (int)            Optional RNG seed; if omitted, one is generated and printed
--out-prefix (str)      Output prefix (PNG, CSV, XLSX)

Example (one line):
`python exp3_random.py --width 10 --height 6 --dot-spacing 0.5 --shape-area 900 --n-quadrats 12 --out-prefix r1`
"""

import argparse
import math
import random
import os
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------ Geometry ------------------------

@dataclass(frozen=True)
class SquareGeom:
    side: float   # meters
    half: float   # meters

def build_square_from_area(area_m2: float) -> SquareGeom:
    if area_m2 <= 0:
        raise ValueError("--shape-area must be > 0 (after unit conversion).")
    side = math.sqrt(area_m2)
    return SquareGeom(side=side, half=side / 2.0)

# ------------------------ Grid & mask ------------------------

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
             centers: np.ndarray,
             sq: SquareGeom,
             out_png: str,
             title: str,
             seed_note: str) -> None:

    fig, ax = plt.subplots(figsize=(10, 10 * (height / max(width, 1e-9))))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Boundary and dots
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, linewidth=1.2))
    ax.scatter(points[:, 0], points[:, 1], s=10, marker='o', linewidths=0, alpha=0.6)

    # Squares
    for (cx, cy) in centers:
        ax.add_patch(plt.Rectangle((cx - sq.half, cy - sq.half), sq.side, sq.side,
                                   facecolor="#9EC9FF", alpha=0.45, linewidth=1.2, edgecolor="black"))

    # Labels
    ax.set_xlabel("Meters (X)")
    ax.set_ylabel("Meters (Y)")
    ax.set_title(title)

    # Legend (single entry)
    side_cm = int(round(sq.side * 100))
    handles = [plt.Line2D([], [], marker='s', linestyle='None', markersize=10,
                          markerfacecolor="#9EC9FF", markeredgecolor="black",
                          label=f"Square ({side_cm}×{side_cm} cm)")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.85, title="Quadrat")

    # Seed note bottom-left (transparent box)
    ax.text(0.001, 0.001, seed_note, transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# ------------------------ Tables ------------------------

def build_table(centers: np.ndarray, sq: SquareGeom, meta: dict) -> pd.DataFrame:
    rows = []
    area = sq.side * sq.side
    for i, (cx, cy) in enumerate(centers, start=1):
        rows.append({
            "quadrat_id": i,
            "center_x_m": round(float(cx), 6),
            "center_y_m": round(float(cy), 6),
            "square_side_m": round(sq.side, 6),
            "square_area_m2": round(area, 6),
            "width_m": meta["width_m"],
            "height_m": meta["height_m"],
            "dot_spacing_m": meta["dot_spacing_m"],
            "seed": meta["seed"],
            "out_prefix": meta["out_prefix"],
        })
    return pd.DataFrame(rows)

def write_tables(df: pd.DataFrame, out_csv: str, out_xlsx: str) -> None:
    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="placements")
        xw.sheets["placements"].freeze_panes(1, 0)

# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Experiment C (Phase 1): square quadrats at randomized dots.")
    p.add_argument("--width", type=float, required=True, help="Study area width in meters")
    p.add_argument("--height", type=float, required=True, help="Study area height in meters")
    p.add_argument("--dot-spacing", type=float, required=True, help="Spacing between dots in meters")
    p.add_argument("--shape-area", type=float, required=True, help="Square quadrat area in cm^2 (single quadrat)")
    p.add_argument("--n-quadrats", type=int, required=True, help="Number of quadrats to place")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--out-prefix", type=str, default="exp3_random", help="Output prefix (PNG, CSV, XLSX)")
    return p.parse_args()

# ------------------------ Main ------------------------

def main():
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Width/height must be > 0.")
    if args.n_quadrats < 1:
        raise SystemExit("--n-quadrats must be >= 1.")
    if args.shape_area <= 0:
        raise SystemExit("--shape-area must be > 0 (in cm^2).")

    # Seed handling (fold to 32-bit), and message
    if args.seed is not None:
        seed = int(args.seed) % (2**32)
        print(f"Seed provided: {seed}")
    else:
        seed = random.randint(0, 2**32 - 1)
        print(f"Random seed generated: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # Geometry from area (cm^2 -> m^2)
    area_cm2 = float(args.shape_area)
    area_m2 = area_cm2 / 1e4
    sq = build_square_from_area(area_m2)

    # Grid and candidates
    pts = dot_grid(args.width, args.height, args.dot_spacing)
    ok = mask_candidates(pts, args.width, args.height, sq)
    candidates = pts[ok]

    if len(candidates) < args.n_quadrats:
        raise SystemExit(
            f"Not enough candidate dots for square area={area_cm2} cm^2 (side={sq.side:.3f} m).\n"
            f"Candidates: {len(candidates)}, requested: {args.n_quadrats}.\n"
            f"Consider reducing --shape-area, increasing --dot-spacing, or enlarging the area."
        )

    # Choose centers without replacement
    idx = np.random.choice(len(candidates), size=args.n_quadrats, replace=False)
    centers = candidates[idx]

    # Meta
    meta = {
        "width_m": args.width,
        "height_m": args.height,
        "dot_spacing_m": args.dot_spacing,
        "seed": seed,
        "out_prefix": args.out_prefix,
    }

    # Build outputs
    df = build_table(centers, sq, meta)

    title = (f"Dot Quadrat Map — Experiment C "
             f"({args.width}m × {args.height}m; dots every {args.dot_spacing}m; "
             f"shape area={area_cm2}cm²; n={args.n_quadrats})")
    seed_note = f"Seed: {seed}"

    # Output directory: ./exp3
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "exp3")
    os.makedirs(out_dir, exist_ok=True)

    # File outputs
    out_png = os.path.join(out_dir, f"{args.out_prefix}.png")
    out_csv = os.path.join(out_dir, f"{args.out_prefix}_placed.csv")
    out_xlsx = os.path.join(out_dir, f"{args.out_prefix}_placed.xlsx")

    # Render & save
    draw_map(args.width, args.height, pts, centers, sq, out_png, title, seed_note)
    write_tables(df, out_csv, out_xlsx)

    print(f"[info] candidates={len(candidates)} / total_dots={len(pts)}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")
    print("Done.")

if __name__ == "__main__":
    main()