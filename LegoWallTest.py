#!/usr/bin/env python3
"""
Test suite for the Lego Brick Builder Simulation.
Comprehensive tests to validate functionality and edge cases.
"""

import time
import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import logging

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LegoWall import (
    Brick, BrickCatalog, WallGenerator, BrickBuilderApp
)


class TestBrick(unittest.TestCase):
    """Test cases for the Brick class."""
    
    def test_brick_creation(self):
        """Test valid brick creation."""
        brick = Brick("TEST123", 2, 1, 0.49, "Red")
        self.assertEqual(brick.catalog_number, "TEST123")
        self.assertEqual(brick.width, 2)
        self.assertEqual(brick.height, 1)
        self.assertEqual(brick.price, 0.49)
        self.assertEqual(brick.color, "Red")
    
    def test_brick_invalid_dimensions(self):
        """Test brick creation with invalid dimensions."""
        with self.assertRaises(ValueError):
            Brick("TEST123", -1, 1, 0.49, "Red")
        
        with self.assertRaises(ValueError):
            Brick("TEST123", 2, 0, 0.49, "Red")
    
    def test_brick_invalid_price(self):
        """Test brick creation with invalid price."""
        with self.assertRaises(ValueError):
            Brick("TEST123", 2, 1, -0.10, "Red")
    
    def test_brick_hashable(self):
        """Test that brick objects are hashable (frozen dataclass)."""
        brick1 = Brick("TEST123", 2, 1, 0.49, "Red")
        brick2 = Brick("TEST123", 2, 1, 0.49, "Red")
        brick_set = {brick1, brick2}
        self.assertEqual(len(brick_set), 1)


class TestBrickCatalog(unittest.TestCase):
    """Test cases for the BrickCatalog class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.catalog = BrickCatalog()
    
    def test_default_bricks_loaded(self):
        """Test that default bricks are loaded."""
        bricks = self.catalog.get_all_bricks()
        self.assertEqual(len(bricks), 2)
        
        # Check yellow brick
        yellow_brick = self.catalog.get_brick("ABCD1234")
        self.assertEqual(yellow_brick.width, 2)
        self.assertEqual(yellow_brick.color, "Yellow")
        
        # Check blue brick
        blue_brick = self.catalog.get_brick("WXYZ1234")
        self.assertEqual(blue_brick.width, 3)
        self.assertEqual(blue_brick.color, "Blue")
    
    def test_add_brick(self):
        """Test adding a new brick to the catalog."""
        new_brick = Brick("NEW123", 4, 1, 0.75, "Green")
        self.catalog.add_brick(new_brick)
        
        retrieved_brick = self.catalog.get_brick("NEW123")
        self.assertEqual(retrieved_brick, new_brick)
    
    def test_get_nonexistent_brick(self):
        """Test getting a non-existent brick."""
        with self.assertRaises(ValueError):
            self.catalog.get_brick("NONEXISTENT")
    
    def test_get_bricks_by_height(self):
        """Test filtering bricks by height."""
        height_1_bricks = self.catalog.get_bricks_by_height(1)
        self.assertEqual(len(height_1_bricks), 2)
        
        height_2_bricks = self.catalog.get_bricks_by_height(2)
        self.assertEqual(len(height_2_bricks), 0)


# class TestWallConfiguration(unittest.TestCase):
#     """Test cases for the WallConfiguration class."""
    
#     def setUp(self):
#         """Set up test fixtures."""
#         self.yellow_brick = Brick("ABCD1234", 2, 1, 0.49, "Yellow")
#         self.blue_brick = Brick("WXYZ1234", 3, 1, 0.55, "Blue")
    
#     def test_wall_configuration_creation(self):
#         """Test wall configuration creation."""
#         # Create a 5-unit wide wall: [2-unit brick][3-unit brick]
#         positions = [(0, self.yellow_brick), (2, self.blue_brick)]
#         config = WallConfiguration(5, positions)
        
#         self.assertEqual(config.width, 5)
#         self.assertEqual(len(config.brick_positions), 2)
#         self.assertEqual(config.crack_positions, {2})  # Crack at position 2
    
#     def test_crack_positions_calculation(self):
#         """Test crack positions calculation."""
#         # Wall: [2][3][2] (width=7)
#         positions = [(0, self.yellow_brick), (2, self.blue_brick), (5, self.yellow_brick)]
#         config = WallConfiguration(7, positions)
        
#         # Cracks at positions 2 and 5
#         self.assertEqual(config.crack_positions, {2, 5})
    
#     def test_valid_next_row(self):
#         """Test validation of next row configuration."""
#         # First row: [2][3] (crack at 2)
#         row1 = WallConfiguration(5, [(0, self.yellow_brick), (2, self.blue_brick)])
        
#         # Second row: [3][2] (crack at 3) - should be valid
#         row2 = WallConfiguration(5, [(0, self.blue_brick), (3, self.yellow_brick)])
        
#         self.assertTrue(row1.is_valid_next_row(row2))
    
#     def test_invalid_next_row(self):
#         """Test invalid next row configuration."""
#         # First row: [2][3] (crack at 2)
#         row1 = WallConfiguration(5, [(0, self.yellow_brick), (2, self.blue_brick)])
        
#         # Second row: [2][3] (crack at 2) - should be invalid (same crack position)
#         row2 = WallConfiguration(5, [(0, self.yellow_brick), (2, self.blue_brick)])
        
#         self.assertFalse(row1.is_valid_next_row(row2))


class TestWallGenerator(unittest.TestCase):
    """Test cases for the WallGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.catalog = BrickCatalog()
        self.generator = WallGenerator(self.catalog)
    
    def test_generate_row_configurations_width_5(self):
        """Test generating row configurations for width 5."""
        configs = self.generator._generate_row_configurations(5)
        self.assertEqual(len(configs), 2)  # [2,3] and [3,2]
        
        # Check that we have both configurations
        config_patterns = []
        for config in configs:
            config_patterns.append(config)
        
        self.assertIn([2, 3], config_patterns)
        self.assertIn([3, 2], config_patterns)
    
    def test_bit_pattern_conversion(self):
        """Test conversion of configurations to bit patterns."""
        # Test [2, 3] configuration (crack at position 2)
        config = [2, 3]
        bit_pattern = self.generator._config_to_bit_pattern(config, 5)
        expected = 1 << 2  # Bit set at position 2
        self.assertEqual(bit_pattern, expected)
        
        # Test [3, 2] configuration (crack at position 3)
        config = [3, 2]
        bit_pattern = self.generator._config_to_bit_pattern(config, 5)
        expected = 1 << 3  # Bit set at position 3
        self.assertEqual(bit_pattern, expected)
    
    def test_count_valid_walls_1x1(self):
        """Test counting valid walls for 1x1 (impossible with available bricks)."""
        result = self.generator.count_valid_walls(1, 1)
        self.assertEqual(result, 0)
    
    def test_count_valid_walls_2x1(self):
        """Test counting valid walls for 2x1."""
        result = self.generator.count_valid_walls(2, 1)
        self.assertEqual(result, 1)  # Only one way: [2]
    
    def test_count_valid_walls_3x1(self):
        """Test counting valid walls for 3x1."""
        result = self.generator.count_valid_walls(3, 1)
        self.assertEqual(result, 1)  # Only one way: [3]
    
    def test_count_valid_walls_5x1(self):
        """Test counting valid walls for 5x1."""
        result = self.generator.count_valid_walls(5, 1)
        self.assertEqual(result, 2)  # Two ways: [2,3] and [3,2]
    
    def test_count_valid_walls_5x2(self):
        """Test counting valid walls for 5x2."""
        result = self.generator.count_valid_walls(5, 2)
        # First row: [2,3] or [3,2]
        # Second row must not have cracks aligned
        # Expected: 2 (each first row has 1 valid second row)
        self.assertEqual(result, 2)
    
    def test_count_valid_walls_9x3(self):
        """Test counting valid walls for 9x3 (known result: 8)."""
        result = self.generator.count_valid_walls(9, 3)
        self.assertEqual(result, 8)
    
    def test_caching_behavior(self):
        """Test that caching improves performance."""
        # First call
        start_time = time.time()
        result1 = self.generator.count_valid_walls(12, 6)
        first_time = time.time() - start_time
        
        # Second call should be faster due to caching
        start_time = time.time()
        result2 = self.generator.count_valid_walls(12, 6)
        second_time = time.time() - start_time
        
        self.assertEqual(result1, result2)
        # Second call should be significantly faster
        self.assertLess(second_time, first_time)


class TestBrickBuilderApp(unittest.TestCase):
    """Test cases for the BrickBuilderApp class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = BrickBuilderApp()
    
    def test_app_initialization(self):
        """Test app initialization."""
        self.assertIsNotNone(self.app.brick_catalog)
        self.assertIsNotNone(self.app.wall_generator)
        self.assertIsNotNone(self.app.logger)
    
    @patch('LegoWall.WallGenerator.count_valid_walls')
    def test_app_run(self, mock_count):
        """Test app run method."""
        mock_count.return_value = 42
        
        result = self.app.run(10, 5)
        
        self.assertEqual(result, 42)
        mock_count.assert_called_once_with(10, 5)
    
    def test_app_run_with_exception(self):
        """Test app run method with exception."""
        with patch.object(self.app.wall_generator, 'count_valid_walls', 
                         side_effect=Exception("Test error")):
            with self.assertRaises(Exception):
                self.app.run(10, 5)

class TestCheapestWall(unittest.TestCase):
    """Test cases for cheapest wall cost computation."""

    def setUp(self):
        self.app = BrickBuilderApp()

    def test_cheapest_wall_cost_5x1(self):
        """Test cheapest wall cost for a 5x1 wall."""
        cost = self.app.run_cheapest_wall(5, 1)
        # Cheapest configuration: [2,3] or [3,2]
        # Yellow = $0.49, Blue = $0.55 → Total = $1.04
        self.assertAlmostEqual(cost, 1.04, places=2)

    def test_cheapest_wall_cost_9x3(self):
        """Test cheapest wall cost for a 9x3 wall."""
        cost = self.app.run_cheapest_wall(9, 3)
        # There are 8 valid configurations, we only check cost is positive
        self.assertAlmostEqual(cost, 5.32, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire system."""
    
    def test_known_results(self):
        """Test against known results."""
        app = BrickBuilderApp()
        
        # Test 9x3 wall (given in problem statement)
        result_9x3 = app.run(9, 3)
        self.assertEqual(result_9x3, 8)
        
        # Test smaller cases we can verify manually
        result_5x2 = app.run(5, 2)
        self.assertEqual(result_5x2, 2)
    
    def test_edge_cases(self):
        """Test edge cases."""
        app = BrickBuilderApp()
        
        # Width too small for any brick
        result = app.run(1, 1)
        self.assertEqual(result, 0)
        
        # Exact brick widths
        result_2x1 = app.run(2, 1)
        self.assertEqual(result_2x1, 1)
        
        result_3x1 = app.run(3, 1)
        self.assertEqual(result_3x1, 1)


def run_performance_tests():
    """Run performance tests for larger wall sizes."""
    print("Running performance tests...")
    
    app = BrickBuilderApp()
    
    # Disable logging for performance tests
    logging.getLogger().setLevel(logging.ERROR)
    
    test_cases = [
        (9, 3, 8),      # Known result
        (15, 15, None), # Lego requirement
        (34, 12, None), # Lego requirement
        (20, 8, None),  # Performance test
        (25, 6, None),  # Performance test
    ]
    
    print("\nPerformance Test Results:")
    print("=" * 50)
    
    for width, height, expected in test_cases:
        print(f"\nTesting {width}x{height} wall:")
        try:
            import time
            start_time = time.time()
            result = app.run(width, height)
            end_time = time.time()
            
            status = "✓" if expected is None or result == expected else "✗"
            print(f"{status} Result: {result:,} (computed in {end_time - start_time:.3f} seconds)")
            
            if expected is not None and result != expected:
                print(f"  Expected: {expected}")
                
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "="*50)
    run_performance_tests()