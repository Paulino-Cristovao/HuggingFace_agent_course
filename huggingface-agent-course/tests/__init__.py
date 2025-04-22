"""Test package initialization for the Hugging Face agent course.

This module contains a dummy test case to ensure the test suite is set up correctly.
"""

import unittest


class DummyTest(unittest.TestCase):
    """A dummy test class to validate the test framework setup."""
    
    def test_addition(self):
        """Test basic addition to ensure arithmetic operations work as expected."""
        self.assertEqual(1 + 1, 2, "1 + 1 should equal 2")
    
    def test_instance_check(self):
        """Test type checking to ensure the value is of the expected type."""
        value = "test"
        self.assertIsInstance(value, str, "Value should be a string")


if __name__ == "__main__":
    # Run the test suite
    unittest.main()