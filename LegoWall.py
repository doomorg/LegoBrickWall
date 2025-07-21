#!/usr/bin/env python3
"""
Lego Brick Wall Builder Simulation

A flexible and extensible library for calculating valid Lego brick wall configurations.
This simulation uses dynamic programming and bit manipulation for efficient computation
of wall patterns that avoid vertical crack alignment.

Key Features:
- Dynamic programming optimization for performance
- Bit manipulation for pattern matching
- Extensible brick catalog system
- Cost optimization capabilities
- Color and dimension filtering
- Comprehensive validation

"""

import argparse
import sys
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import time
import re

# Default brick configurations - can be easily modified or extended
DEFAULT_BRICK_CONFIGS = [
    {
        "catalog_number": "ABCD1234",
        "width": 2,
        "height": 1,
        "price": 0.49,
        "color": "Yellow"
    },
    {
        "catalog_number": "WXYZ1234", 
        "width": 3,
        "height": 1,
        "price": 0.55,
        "color": "Blue"
    }
]

# Validation patterns
CATALOG_NUMBER_PATTERN = re.compile(r'^[A-Z0-9]{4,12}$')
VALID_COLORS = {
    'Red', 'Blue', 'Yellow', 'Green', 'Orange', 'Purple', 'Pink', 
    'Brown', 'Black', 'White', 'Gray', 'Light Blue', 'Dark Green'
}


@dataclass(frozen=True)
class Brick:
    """Represents a brick with its properties."""
    catalog_number: str
    width: int
    height: int
    price: float
    color: str
    
    def __post_init__(self):
        """Validate brick properties."""
        
        # Validate width and height
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Brick dimensions must be positive: width={self.width}, height={self.height}")
        
        # Validate price
        if self.price < 0:
            raise ValueError(f"Brick price cannot be negative: price={self.price}")
        
        # Validate catalog number
        if not CATALOG_NUMBER_PATTERN.match(self.catalog_number):
            raise ValueError(
                f"Invalid catalog number format '{self.catalog_number}'. "
                f"Must be 4-12 alphanumeric characters (A-Z, 0-9)"
            )
            
        # Validate colour
        if self.color not in VALID_COLORS:
            raise ValueError(
                f"Invalid color '{self.color}'. "
                f"Valid colors: {', '.join(sorted(VALID_COLORS))}"
            )
        


class BrickCatalog:
    """Manages the catalog of available bricks."""
    
    def __init__(self):
        self._bricks: Dict[str, Brick] = {}
        self._initialize_default_bricks()
    
    def _initialize_default_bricks(self):
        """Initialize with default Lego brick types."""
        
        for config in DEFAULT_BRICK_CONFIGS:
            brick = Brick(**config)
            self.add_brick(brick)
    
    def add_brick(self, brick: Brick):
        """Add a brick to the catalog."""
        
        self._bricks[brick.catalog_number] = brick
    
    def get_brick(self, catalog_number: str) -> Brick:
        """Get a brick by catalog number."""
        
        if catalog_number not in self._bricks:
            raise ValueError(f"Brick not found: {catalog_number}")
        return self._bricks[catalog_number]
    
    def get_all_bricks(self) -> List[Brick]:
        """Get all available bricks."""
        
        return list(self._bricks.values())
      
    def get_bricks_by_color(self, allowed_colors: Optional[Set[str]] = None) -> List[Brick]:
        """Get bricks filtered by allowed colors."""
        if allowed_colors is None:
            return self.get_all_bricks()
        return [brick for brick in self._bricks.values() if brick.color in allowed_colors]
    
    def get_bricks_by_height(self, height: int, allowed_colors: Optional[Set[str]] = None) -> List[Brick]:
        """Get all bricks with specified height and allowed colors."""
        return [brick for brick in self.get_bricks_by_color(allowed_colors) if brick.height == height]
    
    def get_brick_widths(self, height: int = 1, allowed_colors: Optional[Set[str]] = None) -> List[int]:
        """Get sorted list of brick widths for given height and allowed colors."""
        return sorted([brick.width for brick in self.get_bricks_by_height(height, allowed_colors)])
      
    


class WallGenerator:
    """
    Wall Generator using dynamic programming and bit manipulation:
    - Bit manipulation for pattern matching
    - Cached computations for efficiency
    
    The algorithm works by:
    - Generating all possible row configurations for the wall width
    - Converting each configuration to a bit pattern representing crack positions
    - Using DP to count valid combinations where no cracks align vertically
    """
    
    def __init__(self, brick_catalog: BrickCatalog):
        self.brick_catalog = brick_catalog
        self._price_lookup: Dict[int, float] = self._precompute_cheapest_prices()
        self.logger = logging.getLogger(__name__)
        
        # Caches for performance
        self._row_configs_cache: Dict[int, List[List[int]]] = {}
        self._bit_patterns_cache: Dict[int, List[int]] = {}
        self._compatibility_cache: Dict[Tuple[int, int], List[List[bool]]] = {}
        
    def _generate_row_configurations(self, width: int) -> List[List[int]]:
        """
        Generate all possible row configurations using dynamic programming.
        
        This method finds all ways to fill a row of given width using available
        brick widths. It uses backtracking to explore all valid combinations.
        
        Args:
            width: The target width to fill
            
        Returns:
            List of configurations, where each configuration is a list of brick widths
        """
        
        if width in self._row_configs_cache:
            return self._row_configs_cache[width]
        
        brick_widths = self.brick_catalog.get_brick_widths(height=1)
        configurations = []
        
        def backtrack(remaining_width: int, current_config: List[int]):
            """
            Recursive backtracking to find all valid configurations.
            
            Args:
                remaining_width: How much width is left to fill
                current_config: Current partial configuration being built
            """
            
            if remaining_width == 0:
                configurations.append(current_config.copy())
                return
            
            # Try each available brick width
            for brick_width in brick_widths:
                if remaining_width >= brick_width:
                    current_config.append(brick_width)
                    backtrack(remaining_width - brick_width, current_config)
                    current_config.pop()
        
        backtrack(width, [])
        self._row_configs_cache[width] = configurations
        return configurations
    
    def _config_to_bit_pattern(self, config: List[int], width: int) -> int:
        """
        Convert row configuration to bit pattern representing crack positions.
        
        Each bit position represents a potential crack location. A bit is set to 1
        if there's a crack at that position.
        
        Args:
            config: List of brick widths in the row
            width: Total width of the row
            
        Returns:
            Integer where each bit represents a crack position
        """
        
        bit_pattern = 0
        position = 0
        
        for brick_width in config:
            position += brick_width
            if position < width:  # Don't mark crack at the end
                bit_pattern |= (1 << position)
        
        return bit_pattern
    
    def _get_bit_patterns(self, width: int) -> List[int]:
        """Get bit patterns for all row configurations."""
        
        if width in self._bit_patterns_cache:
            return self._bit_patterns_cache[width]
        
        configs = self._generate_row_configurations(width)
        bit_patterns = [self._config_to_bit_pattern(config, width) for config in configs]
        
        self._bit_patterns_cache[width] = bit_patterns
        return bit_patterns
    
    def _count_walls_dp_approach(self, width: int, height: int) -> int:
        """
        Core dynamic programming algorithm for counting valid walls.
        
        This method uses a recursive approach with memoization to count all
        valid wall configurations. The key insight is that wall validity
        depends only on adjacent rows, allowing for optimal substructure.
        
        Args:
            width: Wall width in stud units
            height: Wall height in rows
            
        Returns:
            Total number of valid wall configurations
            
        Algorithm:
            1. Generate all possible bit patterns for rows of given width
            2. Use recursive DP with state (remaining_height, previous_pattern)
            3. For each state, try all compatible current patterns
            4. Two patterns are compatible if they have no overlapping cracks
            5. Base case: height=0 returns 1 (one way to build empty wall)
        """
        
        bit_patterns = self._get_bit_patterns(width)
        
        # DP with bit patterns as states
        memo = {}
        
        def count_recursive(remaining_height: int, prev_pattern: int) -> int:
            """
            Recursive helper function with memoization.
            
            Args:
                remaining_height: Number of rows left to place
                prev_pattern: Bit pattern of the previously placed row
                
            Returns:
                Number of valid ways to complete the wall from this state
            """
            
            if remaining_height == 0:
                return 1
            
            state_key = (remaining_height, prev_pattern)
            if state_key in memo:
                return memo[state_key]
            
            total = 0
            for current_pattern in bit_patterns:
                if (prev_pattern & current_pattern) == 0:
                    total += count_recursive(remaining_height - 1, current_pattern)
            
            memo[state_key] = total
            return total
        
        # Start with all possible first rows
        result = 0
        for pattern in bit_patterns:
            result += count_recursive(height - 1, pattern)
        
        return result
    
    def count_valid_walls(self, width: int, height: int) -> int:
        """Count valid wall configurations."""
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Wall dimensions must be positive: width={width}, height={height}")
        
        # Check if wall is buildable
        min_brick_width = min(self.brick_catalog.get_brick_widths(height=1))
        if width < min_brick_width:
            return 0
        
        self.logger.info(f"Calculating valid walls for {width}x{height}")
        start_time = time.time()
        
        # Execute optimized algorithm
        result = self._count_walls_dp_approach(width, height)
        
        end_time = time.time()
        self.logger.info(f"Calculation completed in {end_time - start_time:.3f} seconds")
        
        return result
    
    def generate_wall_configurations(self, width: int, height: int) -> List[List[List[int]]]:
        """
        Generate all valid wall configurations explicitly.
        
        This method generates the actual wall configurations rather than just
        counting them. Useful for small problems where we want to visualize
        all possible solutions.
        
        Returns:
            List of walls, where each wall is a list of rows, and each row
            is a list of brick widths.
            
        Note: This method has exponential complexity and should only be used
        for small wall dimensions to avoid memory/performance issues.
        """
        
        row_configs = self._generate_row_configurations(width)
        bit_patterns = [self._config_to_bit_pattern(config, width) for config in row_configs]
        config_bit_pairs = list(zip(row_configs, bit_patterns))
        valid_walls = []

        def backtrack(level: int, prev_bit: int, current_wall: List[List[int]]):
            """
            Recursive backtracking to generate all valid wall configurations.
            
            Args:
                level: Current row being filled (0 to height-1)
                prev_bit: Bit pattern of the previous row
                current_wall: Wall configuration being built
            """
            
            if level == height:
                valid_walls.append([row.copy() for row in current_wall])
                return
            for config, bit in config_bit_pairs:
                if level == 0 or (prev_bit & bit) == 0:
                    current_wall.append(config)
                    backtrack(level + 1, bit, current_wall)
                    current_wall.pop()

        backtrack(0, 0, [])
        return valid_walls
    
    def _precompute_cheapest_prices(self) -> Dict[int, float]:
        """Precompute the cheapest price for each brick width."""
        lookup = {}
        for brick in self.brick_catalog.get_bricks_by_height(1):
            if brick.width not in lookup or brick.price < lookup[brick.width]:
                lookup[brick.width] = brick.price
        return lookup

    def compute_wall_cost(self, wall: List[List[int]]) -> float:
        """Compute total cost of a given wall configuration using cached prices."""
        total = 0.0
        for row in wall:
            for width in row:
                if width not in self._price_lookup:
                    raise ValueError(f"No matching brick of width {width}")
                total += self._price_lookup[width]
        return total

    def generate_wall_configurations_with_costs(self, width: int, height: int) -> List[Tuple[List[List[int]], float]]:
        """Return all valid wall configurations with their total cost."""
        walls = self.generate_wall_configurations(width, height)
        return [(w, self.compute_wall_cost(w)) for w in walls]

class BrickBuilderApp:
    """
    Main application class providing a comprehensive interface to the brick builder simulation.
    
    This class orchestrates all components of the system and provides both programmatic
    and command-line interfaces. It's designed to be easily extensible for future
    enhancements like:
    - Cost optimization algorithms
    - Color pattern constraints  
    - Advanced wall generation strategies
    - Export capabilities for wall plans
    """
    
    def __init__(self):
        """Initialize the Brick Builder application."""
        
        self.brick_catalog = BrickCatalog()
        self.wall_generator = WallGenerator(self.brick_catalog)
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
      
    def run_cheapest_wall(self, width: int, height: int, allowed_colors: Optional[Set[str]] = None):
        self.logger.info("Finding cheapest configuration")
        self.logger.info(f"Allowed colors: {allowed_colors if allowed_colors else 'All'}")
        
        self.wall_generator.brick_catalog._bricks = {
            k: b for k, b in self.wall_generator.brick_catalog._bricks.items() if allowed_colors is None or b.color in allowed_colors
        }
        self.wall_generator._price_lookup = self.wall_generator._precompute_cheapest_prices()
        walls_with_costs = self.wall_generator.generate_wall_configurations_with_costs(width, height)
        if not walls_with_costs:
            print("No configurations found.")
            return 0.0
        wall, cost = min(walls_with_costs, key=lambda x: x[1])
        print("\n--- Cheapest Wall Configuration ---")
        for row in wall:
            print('|'.join(['■' * w for w in row]))
        print(f"Total cost: ${cost:.2f}")
        return cost
    
    def run(self, wall_width: int, wall_height: int, cheapest_mode: bool = False, color_filter: Optional[List[str]] = None) -> int:
        """
        Run the brick builder simulation.
        
        Args:
            wall_width: Width of the wall
            wall_height: Height of the wall
            
        Returns:
            Number of unique valid wall configurations
        """
        try:
            self.logger.info(f"Starting optimized brick builder for {wall_width}x{wall_height} wall")
            allowed_colors = set(color_filter) if color_filter else None
            self.wall_generator.brick_catalog._bricks = {
                k: b for k, b in self.wall_generator.brick_catalog._bricks.items()
                if allowed_colors is None or b.color in allowed_colors
            }
            self.wall_generator._row_configs_cache.clear()
            self.wall_generator._bit_patterns_cache.clear()
            self.wall_generator._price_lookup = self.wall_generator._precompute_cheapest_prices()
            allowed_colors = set(color_filter) if color_filter else None
            if cheapest_mode:
                return self.run_cheapest_wall(wall_width, wall_height, allowed_colors)
  
            # Display available bricks
            bricks = self.brick_catalog.get_all_bricks()
            self.logger.info(f"Available bricks: {len(bricks)}")
            for brick in bricks:
                self.logger.info(f"  - {brick.color} ({brick.catalog_number}): "
                               f"{brick.width}x{brick.height} units, ${brick.price}")
            if len(bricks) == 0:
                self.logger.warning("No bricks with specified colours available in the catalog.")
                return 0
              
            # Calculate unique wall configurations
            result = self.wall_generator.count_valid_walls(wall_width, wall_height)
            
            self.logger.info(f"Total unique valid wall configurations: {result}")
            
            # If result is small, generate and display wall configurations
            if result < 10:
              walls = self.wall_generator.generate_wall_configurations(wall_width, wall_height)
              print("\n--- Wall Configurations (brick widths per row) ---")
              for i, wall in enumerate(walls):
                  print(f"\nConfiguration {i+1}:")
                  for row in wall:
                      line = '|'.join(['■' * w for w in row])
                      print(line)
            return result
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            raise


def main():
    """Main entry point for the command line application."""
    parser = argparse.ArgumentParser(
        description="Optimized Lego Brick Builder - Calculate unique valid wall configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument(
        'wall_width',
        type=int,
        help='Width of the wall in units'
    )
    
    parser.add_argument(
        'wall_height',
        type=int,
        help='Height of the wall in units'
    )
    
    parser.add_argument(
        '--cheapest',
        action='store_true',
        help='Find and print the cheapest valid wall configuration'
    )
    
    parser.add_argument(
        '--colors',
        nargs='+',
        help='Specify allowed brick colors (e.g., --colors Red Blue Yellow)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.wall_width <= 0 or args.wall_height <= 0:
        print("Error: Wall dimensions must be positive integers", file=sys.stderr)
        sys.exit(1)
    
    try:
        app = BrickBuilderApp()
        result = app.run(args.wall_width, args.wall_height, cheapest_mode=args.cheapest, color_filter=args.colors)
        if not args.cheapest:
          print(result)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()