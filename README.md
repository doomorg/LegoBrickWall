# LegoBrickWall

A flexible and extensible Python library for calculating valid Lego brick wall configurations. This simulation uses dynamic programming and bit manipulation to efficiently compute wall patterns that avoid vertical crack alignment.

# Features

- Dynamic Programming Optimization: Efficient algorithm for large wall calculations
- Bit Manipulation: Fast pattern matching for crack alignment detection
- Extensible Brick Catalog: Easy to add custom bricks and filter by various criteria
- Cost Optimization: Find most cost-effective wall configurations
  - Significant run time for larger wall sizes with > 1,000,000 configurations
- Colour Filtering: Restrict calculations to specific brick types
- Comprehensive Validation: Robust error handling and input validation
- CLI & Library Interface: Use as standalone tool or import as library

# Installation

# Clone the repository
git clone git@github.com:doomorg/LegoBrickWall.git
cd LegoBrickWall

# No external dependencies required - uses only Python standard library
python brick_builder.py --help

# Basic Command Line Usage
python brick_builder.py 6 3

# Calculate with verbose output
python brick_builder.py 6 3 --verbose

# Filter by colour
python brick_builder.py 8 4 --colors Blue Yellow

# Find cheapest wall configuration
python brick_builder.py 10 5 --cheapest

# Extensions
Possible extensions include:
- Adding more bricks with more colours. This will make colour filtering more useful.
- Adding bricks with different heights
- Finding number of walls that satisfy a maximum cost