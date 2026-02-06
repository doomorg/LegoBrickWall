# Lego Brick Wall Builder

A Python library and CLI for counting and generating valid Lego brick wall configurations. Walls are built from rows of bricks (widths 2 and 3 by default); the only rule is **no vertical crack alignment** between adjacent rows (no continuous vertical line where bricks meet).

## Features

- **Efficient counting** — Iterative dynamic programming with precomputed row compatibility; each row height is processed once with O(1) compatibility lookup.
- **Crack-set model** — Rows are represented as sets of crack positions; compatibility is precomputed so valid neighbour rows are found quickly.
- **Extensible catalog** — Add bricks, filter by colour and dimensions; default bricks are 2×1 and 3×1 (yellow and blue).
- **Cost optimisation** — Find the cheapest valid wall configuration for given dimensions and optional colour filter.
- **CLI and library** — Run from the command line or import `BrickCatalog`, `WallGenerator`, and `BrickBuilderApp` in your own code.

## Requirements

- Python 3.8+

No external dependencies.

## Installation

Clone the repository:

```bash
git clone git@github.com:doomorg/LegoBrickWall.git
cd LegoBrickWall
```

## Usage

### Command line

Count valid configurations for a wall of given width × height (in studs/rows):

```bash
python LegoWall.py <width> <height>
```

Examples:

```bash
# Count valid 6×3 walls
python LegoWall.py 6 3

# Verbose logging
python LegoWall.py 6 3 --verbose

# Restrict to specific colours
python LegoWall.py 8 4 --colors Blue Yellow

# Print the cheapest valid wall and its cost
python LegoWall.py 10 5 --cheapest
```

### As a library

```python
from LegoWall import BrickCatalog, WallGenerator, BrickBuilderApp

catalog = BrickCatalog()
generator = WallGenerator(catalog)

# Count valid configurations
count = generator.count_valid_walls(9, 3)  # e.g. 8

# Generate all configurations (use with care for large counts)
walls = generator.generate_wall_configurations(5, 2)
```

## Algorithm (summary)

1. **Row configurations** — For a given width, generate all ways to fill one row using available brick widths (e.g. 2 and 3).
2. **Crack sets** — For each row, compute the set of “crack” positions (boundaries between bricks). Two rows can be stacked only if their crack sets are disjoint (no shared vertical crack).
3. **Precomputed compatibility** — For each row pattern, store the list of row indices that can sit next to it (disjoint crack sets).
4. **Iterative DP** — For height 1, each pattern has count 1. For each extra row, `new_dp[i] = sum(dp[j] for j in compatible_with[i])`. The total count is the sum of the final `dp`.

For more detail, see `ALGORITHM_AND_TESTING_EXPLANATION.md`.

## Tests

Run the test suite (including performance checks):

```bash
python LegoWallTest.py
```

Or with unittest:

```bash
python -m unittest LegoWallTest -v
```

## Project layout

| File | Purpose |
|------|--------|
| `LegoWall.py` | Main library and CLI: `Brick`, `BrickCatalog`, `WallGenerator`, `BrickBuilderApp`. |
| `LegoWallTest.py` | Unit and integration tests; run for validation and basic performance checks. |
| `ALGORITHM_AND_TESTING_EXPLANATION.md` | Detailed algorithm description and testing notes. |

## Possible extensions

- More brick sizes and colours to make colour filtering and cost optimisation more interesting.
- Support for bricks with height > 1 (multi-row bricks).
- Count or enumerate walls under a maximum cost.
- Export wall plans (e.g. visual or BOM) for building.
