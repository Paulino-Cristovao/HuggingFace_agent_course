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
    unittest.main()    """Test package initialization for the Hugging Face agent course.
    
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
        """Agent package initialization."""
        """
    Main module for the Hugging Face agent.
    
    This module provides the core functionality for the agent, including OpenAI API integration.
    """
    
    import os
    import time
    import logging
    from typing import Any, Callable, Optional, TypeVar, cast
    # Install these packages with: pip install python-dotenv openai
    from dotenv import load_dotenv
    from openai import OpenAI, OpenAIError
    
    T = TypeVar("T")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    load_dotenv()
    
    # Constants
    SYSTEM_PROMPT = (
        "You are an AI assistant that answers questions to the best of your ability. "
        "You can use tools to gather information when needed. Always provide clear, "
        "concise, and accurate answers.\n\n"
        "When using tools, follow this process:\n"
        "1. Think about the action you need to take.\n"
        "2. Use the tool by specifying a JSON object with the following structure:\n"
        "    {\n"
        '      "action": "<tool_name>",\n'
        '      "action_input": {<input_parameters>}\n'
        "    }\n"
        "3. Observe the result of the action and decide the next step.\n\n"
        "Example tool usage:\n"
        "{\n"
        '  "action": "get_weather",\n'
        '  "action_input": {"location": "New York"}\n'
        "}\n\n"
        "Always follow this format:\n"
        "- Question: The input question you must answer.\n"
        "- Thought: Your reasoning about the next action.\n"
        "- Action: The JSON object specifying the tool and input.\n"
        "- Observation: The result of the action.\n"
        "- Repeat Thought/Action/Observation as needed.\n"
        "- Conclude with:\n"
        "  Thought: I now know the final answer.\n"
        '  Final Answer: The final answer to the original question.\n\n'
        'Begin your task now. Always use the exact phrase `Final Answer:` when providing a definitive answer.'
    )
    
    
    def retry(
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,),
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Exponential-backoff retry decorator."""
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                delay = 1.0
                last_err = None
                for attempt in range(1, max_retries + 1):
                    try:
                        return fn(*args, **kwargs)
                    except exceptions as err:
                        last_err = err
                        if attempt == max_retries:
                            logger.error("Retry %d/%d failed: %s", attempt, max_retries, err)
                            raise
                        logger.warning(
                            "Retry %d/%d failed: %r, retrying in %.1fs...",
                            attempt,
                            max_retries,
                            err,
                            delay
                        )