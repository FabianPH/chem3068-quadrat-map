# CHEM3068 Dot Quadrat Map Randomizer

Python tools for generating **randomized quadrat maps** used in **CHEM3068 General Ecology**. Each script outputs a **PNG map** visualizing the study area and quadrat placements, as well as **CSV** and **Excel tables** containing coordinate and parameter data. All files are automatically organized into their respective folders: `exp1/`, `exp2/`, and `exp3/`.

## Scripts

- `exp1_shapes.py` — Places three co-centered quadrats (square, circle, triangle) at each randomly selected dot.  
- `exp2_sizes.py` — Uses squares of varying sizes; adjusts the number of quadrats to maintain a constant total sampled area.
- `exp3_random.py` — Randomly places a defined number of square quadrats across the dot grid.

## Requirements

- Python 3.9+
- Libraries:  `numpy`, `matplotlib`, `pandas`, `XlsxWriter`  

See `setup.txt` for environment setup instructions.

## Usage (one-line examples)

```sh
python exp1_shapes.py --width 4 --height 2 --dot-spacing 0.5 --shape-area 120 --n-shapes 3
python exp2_sizes.py --width 10 --height 6 --dot-spacing 0.5 --sizes-cm 15,20,25 --target-area-m2 0.1875
python exp3_random.py --width 10 --height 6 --dot-spacing 0.5 --shape-area 900 --n-quadrats 12
