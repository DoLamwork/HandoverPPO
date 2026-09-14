"""Include the subordinate simulator suite in the principal project's tests."""

from pathlib import Path
import unittest


def load_tests(loader, tests, pattern):
    directory = Path(__file__).resolve().parents[1] / "data_generator" / "tests"
    simulator_loader = unittest.TestLoader()
    return simulator_loader.discover(str(directory), pattern="test_*.py")
