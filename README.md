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

# Basic Command Line Usage
python3 LegoBrickWall.py 6 3

# Calculate with verbose output
python3 LegoBrickWall.py 6 3 --verbose

# Filter by colour
python3 LegoBrickWall.py 8 4 --colors Blue Yellow

# Find cheapest wall configuration
python3 LegoBrickWall.py 10 5 --cheapest

# Extensions
Possible extensions include:
- Adding more bricks with more colours. This will make colour filtering more useful.
- Adding bricks with different heights
- Finding number of walls that satisfy a maximum cost
