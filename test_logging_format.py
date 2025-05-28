#!/usr/bin/env python3
"""Test script to verify the new logging format."""

import os

os.environ["HYPERSLOTH_LOCAL_RANK"] = "0"

from HyperSloth.logging_config import get_hypersloth_logger


def test_function():
    """Test function to check logging format."""
    logger = get_hypersloth_logger(gpu_id="0", log_level="INFO")

    logger.log_success("Testing new logging format!")
    logger.log_warning("This is a warning message")
    logger.log_error("This is an error message")

    # Test regular logger methods
    logger.logger.info("Regular info message")
    logger.logger.debug("Debug message")


if __name__ == "__main__":
    test_function()
