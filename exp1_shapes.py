#!/usr/bin/env python3
"""
Dot Quadrat Map — Experiment 1 (Multiple Shapes)
------------------------------------------------
Randomly choose dots on a lattice and, at each chosen dot, place THREE co-centered quadrats
(one square, one circle, one equilateral triangle), all with the SAME area.

CLI
---
--width (m)            Study area width
--height (m)           Study area height
--dot-spacing (m)      Distance between dots
--shape-area (cm^2)    Area of ONE quadrat (for each shape) in cm^2
--n-shapes             Number of DOTS to randomly select (each dot has 3 quadrats)
--seed                 Optional RNG seed; if not given, one is generated and printed
--out-prefix           Output prefix (PNG, CSV, XLSX)

Example (one line):
`python exp1_shapes.py --width 4 --height 2 --dot-spacing 0.25 --shape-area 220 --n-shapes 5 --out-prefix r1`
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

# ------------------------ Geometry helpers ------------------------

@dataclass(frozen=True)
class ShapeGeom:
    shape: str
    side: float = 0.0
    radius: float = 0.0
    tri_height: float = 0.0
    half_width: float = 0.0
    up_extent: float = 0.0
    down_extent: float = 0.0

def build_shape_from_area(shape: str, area_m2: float) -> ShapeGeom:
    if area_m2 <= 0:
        raise ValueError("--shape-area must be > 0 (after unit conversion).")
    shape = shape.lower()
    if shape == "square":
        side = math.sqrt(area_m2)
        half = side / 2.0
        return ShapeGeom(shape="square", side=side,
                         half_width=half, up_extent=half, down_extent=half)
    elif shape == "circle":
        radius = math.sqrt(area_m2 / math.pi)
        return ShapeGeom(shape="circle", radius=radius,
                         half_width=radius, up_extent=radius, down_extent=radius)
    elif shape == "triangle":
        # Equilateral triangle
        side = math.sqrt(4.0 * area_m2 / math.sqrt(3.0))
        h = (math.sqrt(3.0) / 2.0) * side
        up = (2.0 / 3.0) * h
        down = (1.0 / 3.0) * h
        half_w = side / 2.0
        return ShapeGeom(shape="triangle", side=side, tri_height=h,
                         half_width=half_w, up_extent=up, down_extent=down)
    else:
        raise ValueError("Unsupported shape. Use square|circle|triangle.")

def build_all_geoms_same_area(area_m2: float) -> Dict[str, ShapeGeom]:
    return {
        "square":   build_shape_from_area("square", area_m2),
        "circle":   build_shape_from_area("circle", area_m2),
        "triangle": build_shape_from_area("triangle", area_m2),
    }

def dot_grid(width: float, height: float, spacing: float) -> np.ndarray:
    if spacing <= 0:
        raise ValueError("--dot-spacing must be > 0.")
    xs = np.arange(0.0, width + 1e-9, spacing)
    ys = np.arange(0.0, height + 1e-9, spacing)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xv.ravel(), yv.ravel()])

def mask_candidates_multi(points: np.ndarray, width: float, height: float, geoms: Dict[str, ShapeGeom]) -> np.ndarray:
    # Require that ALL three shapes would fit at a dot
    half_w_max = max(g.half_width for g in geoms.values())
    up_max = max(g.up_extent for g in geoms.values())
    down_max = max(g.down_extent for g in geoms.values())
    x, y = points[:, 0], points[:, 1]
    ok = (
        (x - half_w_max >= 0.0) &
        (x + half_w_max <= width) &
        (y - down_max >= 0.0) &
        (y + up_max <= height)
    )
    return ok

# ------------------------ Drawing ------------------------

def draw_map_exp1(width: float,
                  height: float,
                  points: np.ndarray,
                  chosen: np.ndarray,
                  geoms: Dict[str, ShapeGeom],
                  out_png: str,
                  title: str,
                  note_lines: List[str]) -> None:
    COLORS = {
        "square":   "#87CEFA",  # light blue
        "triangle": "#FFF9B1",  # light yellow
        "circle":   "#FF9999",  # light red
    }

    fig, ax = plt.subplots(figsize=(10, 10 * (height / max(width, 1e-9))))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Boundary and all dots
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, linewidth=1.2))
    ax.scatter(points[:, 0], points[:, 1], s=10, marker='o', linewidths=0, alpha=0.6)

    # Three shapes at each chosen dot
    for (cx, cy) in chosen:
        # square
        gs = geoms["square"]
        half = gs.side / 2.0
        ax.add_patch(plt.Rectangle((cx - half, cy - half), gs.side, gs.side,
                                   facecolor=COLORS["square"], alpha=0.45, linewidth=1.2, edgecolor="black"))
        # circle
        gc = geoms["circle"]
        ax.add_patch(plt.Circle((cx, cy), radius=gc.radius,
                                facecolor=COLORS["circle"], alpha=0.45, linewidth=1.2, edgecolor="black"))
        # triangle
        gt = geoms["triangle"]
        h = gt.tri_height
        up, down, s = (2.0 / 3.0) * h, (1.0 / 3.0) * h, gt.side
        base_left, base_right, apex = (cx - s / 2.0, cy - down), (cx + s / 2.0, cy - down), (cx, cy + up)
        ax.add_patch(plt.Polygon([base_left, base_right, apex],
                                 closed=True, facecolor=COLORS["triangle"], alpha=0.45, linewidth=1.2, edgecolor="black"))

    ax.set_xlabel("Meters (X)")
    ax.set_ylabel("Meters (Y)")
    ax.set_title(title)

    # Legend
    handles = [
        plt.Line2D([], [], marker='s', linestyle='None', markersize=10,
                   markerfacecolor=COLORS["square"], markeredgecolor="black", label="Square"),
        plt.Line2D([], [], marker='o', linestyle='None', markersize=10,
                   markerfacecolor=COLORS["circle"], markeredgecolor="black", label="Circle"),
        plt.Line2D([], [], marker=(3, 0, 0), linestyle='None', markersize=12,
                   markerfacecolor=COLORS["triangle"], markeredgecolor="black", label="Triangle"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.85, title="Quadrat Type")

    # Seed note bottom-left
    note = "\n".join(note_lines)
    ax.text(0.001, 0.001, note, transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# ------------------------ CSV / XLSX ------------------------

def write_tables_exp1(chosen: np.ndarray,
                      geoms: Dict[str, ShapeGeom],
                      area_cm2: float,
                      meta: dict,
                      out_csv: str,
                      out_xlsx: str) -> None:
    rows = []
    for i, (cx, cy) in enumerate(chosen, start=1):
        replicate_id = (i - 1) // 3 + 1  # retained logic; can be ignored downstream if not needed
        for shape_name, geom in geoms.items():
            row = {
                "experiment": "1",
                "replicate_id": replicate_id,
                "dot_index": i,
                "center_x_m": round(float(cx), 6),
                "center_y_m": round(float(cy), 6),
                "shape": shape_name,
                "shape_area_cm2": round(area_cm2, 6),
                "shape_area_m2": round(area_cm2 / 1e4, 9),
                "square_side_m": round(geom.side, 6) if shape_name == "square" else None,
                "circle_radius_m": round(geom.radius, 6) if shape_name == "circle" else None,
                "triangle_side_m": round(geom.side, 6) if shape_name == "triangle" else None,
                "triangle_height_m": round(geom.tri_height, 6) if shape_name == "triangle" else None,
                "width_m": meta["width_m"],
                "height_m": meta["height_m"],
                "dot_spacing_m": meta["dot_spacing_m"],
                "seed": meta["seed"],
                "out_prefix": meta["out_prefix"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="placements")
        xw.sheets["placements"].freeze_panes(1, 0)

# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1: three co-centered shapes per chosen dot (same area per shape).")
    p.add_argument("--width", type=float, required=True, help="Study area width in meters")
    p.add_argument("--height", type=float, required=True, help="Study area height in meters")
    p.add_argument("--dot-spacing", type=float, required=True, help="Spacing between dots in meters")
    p.add_argument("--shape-area", type=float, required=True, help="Area of a single quadrat in cm^2 (applies to each shape)")
    p.add_argument("--n-shapes", type=int, required=True, help="Number of DOTS to randomly select (each dot has 3 quadrats)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--out-prefix", type=str, default="exp1_shapes", help="Output prefix (PNG, CSV, XLSX)")
    return p.parse_args()

# ------------------------ Main ------------------------

def main():
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Width/height must be > 0.")
    if args.n_shapes < 1:
        raise SystemExit("--n-shapes must be >= 1")
    if args.shape_area <= 0:
        raise SystemExit("--shape-area must be > 0 (in cm^2).")

    # Seed handling
    if args.seed is not None:
        seed = int(args.seed) % (2**32)
        print(f"Seed provided: {seed}")
    else:
        seed = random.randint(0, 2**32 - 1)
        print(f"Random seed generated: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # Geometry (cm^2 -> m^2)
    area_cm2 = float(args.shape_area)
    area_m2 = area_cm2 / 1e4
    geoms = build_all_geoms_same_area(area_m2)

    # Grid and candidates
    pts = dot_grid(args.width, args.height, args.dot_spacing)
    ok_mask = mask_candidates_multi(pts, args.width, args.height, geoms)
    candidates = pts[ok_mask]

    if len(candidates) < args.n_shapes:
        raise SystemExit(
            f"Not enough candidate dots where ALL shapes fit fully inside the area.\n"
            f"Candidates available: {len(candidates)}, requested dots: {args.n_shapes}.\n"
            f"Consider reducing --shape-area, increasing --dot-spacing, or enlarging the study area."
        )

    # Choose dots without replacement
    select_idx = np.random.choice(len(candidates), size=args.n_shapes, replace=False)
    chosen = candidates[select_idx]

    # Metadata
    meta = {
        "width_m": args.width,
        "height_m": args.height,
        "dot_spacing_m": args.dot_spacing,
        "seed": seed,
        "out_prefix": args.out_prefix,
        "experiment": "1",
    }

    # Title
    title = (
        f"Dot Quadrat Map — Experiment 1 "
        f"({args.width}m × {args.height}m; dots every {args.dot_spacing}m; "
        f"shape area={area_cm2}cm²; n={args.n_shapes})"
    )

    # Bottom-left note = seed only
    note_lines = [f"Seed: {seed}"]

    # Output directory: ./exp1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "exp1")
    os.makedirs(out_dir, exist_ok=True)

    # Outputs
    out_png = os.path.join(out_dir, f"{args.out_prefix}.png")
    out_csv = os.path.join(out_dir, f"{args.out_prefix}_placed.csv")
    out_xlsx = os.path.join(out_dir, f"{args.out_prefix}_placed.xlsx")

    # Render and save
    draw_map_exp1(args.width, args.height, pts, chosen, geoms, out_png, title, note_lines)
    write_tables_exp1(chosen, geoms, area_cm2, meta, out_csv, out_xlsx)

    print(f"[Info] candidates={len(candidates)} / total_dots={len(pts)}")
    print(f"n_shapes={args.n_shapes}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")
    print("Done.")

if __name__ == "__main__":
    main()