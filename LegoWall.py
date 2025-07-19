#!/usr/bin/env python3
"""
Lego Brick Wall Builder Simulation

"""

import argparse
import sys
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time


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
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Brick dimensions must be positive: width={self.width}, height={self.height}")
        if self.price < 0:
            raise ValueError(f"Brick price cannot be negative: price={self.price}")


class BrickCatalog:
    """Manages the catalog of available bricks."""
    
    def __init__(self):
        self._bricks: Dict[str, Brick] = {}
        self._initialize_default_bricks()
    
    def _initialize_default_bricks(self):
        """Initialize with default Lego brick types."""
        self.add_brick(Brick("ABCD1234", 2, 1, 0.49, "Yellow"))
        self.add_brick(Brick("WXYZ1234", 3, 1, 0.55, "Blue"))
    
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
    
    def get_bricks_by_height(self, height: int) -> List[Brick]:
        """Get all bricks with specified height."""
        return [brick for brick in self._bricks.values() if brick.height == height]
    
    def get_brick_widths(self, height: int = 1) -> List[int]:
        """Get sorted list of brick widths for given height."""
        return sorted([brick.width for brick in self.get_bricks_by_height(height)])


class WallGenerator:
    """
    Wall Generator using dynamic programming and bit manipulation:
    - Bit manipulation for pattern matching
    - Cached computations for efficiency
    """
    
    def __init__(self, brick_catalog: BrickCatalog):
        self.brick_catalog = brick_catalog
        self.logger = logging.getLogger(__name__)
        
        # Caches for performance
        self._row_configs_cache: Dict[int, List[List[int]]] = {}
        self._bit_patterns_cache: Dict[int, List[int]] = {}
        self._compatibility_cache: Dict[Tuple[int, int], List[List[bool]]] = {}
        
    def _generate_row_configurations(self, width: int) -> List[List[int]]:
        """Generate all possible row configurations using dynamic programming."""
        if width in self._row_configs_cache:
            return self._row_configs_cache[width]
        
        brick_widths = self.brick_catalog.get_brick_widths(height=1)
        configurations = []
        
        def backtrack(remaining_width: int, current_config: List[int]):
            if remaining_width == 0:
                configurations.append(current_config.copy())
                return
            
            for brick_width in brick_widths:
                if remaining_width >= brick_width:
                    current_config.append(brick_width)
                    backtrack(remaining_width - brick_width, current_config)
                    current_config.pop()
        
        backtrack(width, [])
        self._row_configs_cache[width] = configurations
        return configurations
    
    def _config_to_bit_pattern(self, config: List[int], width: int) -> int:
        """Convert row configuration to bit pattern representing crack positions."""
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
        """Use dynamic programming and recursion with bit manipulation."""
        
        bit_patterns = self._get_bit_patterns(width)
        
        # DP with bit patterns as states
        memo = {}
        
        def count_recursive(remaining_height: int, prev_pattern: int) -> int:
            if remaining_height == 0:
                return 1
            
            if (remaining_height, prev_pattern) in memo:
                return memo[(remaining_height, prev_pattern)]
            
            total = 0
            for current_pattern in bit_patterns:
                # Check compatibility using bit operations
                if (prev_pattern & current_pattern) == 0:
                    total += count_recursive(remaining_height - 1, current_pattern)
            
            memo[(remaining_height, prev_pattern)] = total
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
        
        # Algorithm selection based on problem characteristics
        brick_widths = self.brick_catalog.get_brick_widths(height=1)
        
        # try:
        result = self._count_walls_dp_approach(width, height)
        
        end_time = time.time()
        self.logger.info(f"Calculation completed in {end_time - start_time:.3f} seconds")
        
        return result
    
    def generate_wall_configurations(self, width: int, height: int) -> List[List[List[int]]]:
        """Generate all valid wall configurations (rows of brick widths) with no vertical cracks aligned."""
        row_configs = self._generate_row_configurations(width)
        bit_patterns = [self._config_to_bit_pattern(config, width) for config in row_configs]
        config_bit_pairs = list(zip(row_configs, bit_patterns))
        valid_walls = []

        def backtrack(level: int, prev_bit: int, current_wall: List[List[int]]):
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


class BrickBuilderApp:
    """Main application class for the brick builder simulation."""
    
    def __init__(self):
        self.brick_catalog = BrickCatalog()
        self.wall_generator = WallGenerator(self.brick_catalog)
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run(self, wall_width: int, wall_height: int) -> int:
        """
        Run the optimized brick builder simulation.
        
        Args:
            wall_width: Width of the wall
            wall_height: Height of the wall
            
        Returns:
            Number of unique valid wall configurations
        """
        try:
            self.logger.info(f"Starting optimized brick builder for {wall_width}x{wall_height} wall")
            
            # Display available bricks
            bricks = self.brick_catalog.get_all_bricks()
            self.logger.info(f"Available bricks: {len(bricks)}")
            for brick in bricks:
                self.logger.info(f"  - {brick.color} ({brick.catalog_number}): "
                               f"{brick.width}x{brick.height} units, ${brick.price}")
            
            # Calculate unique wall configurations
            result = self.wall_generator.count_valid_walls(wall_width, wall_height)
            
            self.logger.info(f"Total unique valid wall configurations: {result}")
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
        result = app.run(args.wall_width, args.wall_height)
        print(result)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()