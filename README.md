# LegoBrickWall

A flexible and extensible Python library for calculating valid Lego brick wall configurations. This simulation uses dynamic programming and bit manipulation to efficiently compute wall patterns that avoid vertical crack alignment.

# Features

- Dynamic Programming Optimization: Efficient algorithm for large wall calculations
- Bit Manipulation: Fast pattern matching for crack alignment detection
- Extensible Brick Catalog: Easy to add custom bricks and filter by various criteria
- Cost Optimization: Find most cost-effective brick combinations
- Color & Size Filtering: Restrict calculations to specific brick types
- Comprehensive Validation: Robust error handling and input validation
- CLI & Library Interface: Use as standalone tool or import as library

# Installation

# Clone the repository
git clone git@github.com:doomorg/LegoBrickWall.git
cd LegoBrickWall

# No external dependencies required - uses only Python standard library
python brick_builder.py --help

# Quick Start

# Basic Command Line Usage

# Calculate configurations for a 6x3 wall
python brick_builder.py 6 3

# Calculate with verbose output
python brick_builder.py 6 3 --verbose

# Filter by color
python brick_builder.py 8 4 --color-filter Blue Yellow

# Filter by maximum cost
python brick_builder.py 10 5 --max-cost 0.50