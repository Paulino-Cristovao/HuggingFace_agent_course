"""
Main entrypoint for the Agent CLI application.
"""

import os
import time
import logging
from typing import Any, Callable, Optional, TypeVar, cast

try:
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError("Missing dependency 'python-dotenv'. Please install it.") from e

try:
    from openai import OpenAI, OpenAIError
except ImportError as e:
    raise ImportError("Missing dependency 'openai'. Please install it.") from e

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
    "You can use tools to gather information when needed. "
    "Always provide clear, concise, and accurate answers.\n\n"
    "When using tools, follow this process:\n"
    "1. Think about the action you need to take.\n"
    "2. Use the tool by specifying a JSON object with the following structure:\n"
    "    {\n"
    "      \"action\": \"<tool_name>\",\n"
    "      \"action_input\": {<input_parameters>}\n"
    "    }\n"
    "3. Observe the result of the action and decide the next step.\n\n"
    "Example tool usage:\n"
    "{\n"
    "  \"action\": \"get_weather\",\n"
    "  \"action_input\": {\"location\": \"New York\"}\n"
    "}\n\n"
    "Always follow this format:\n"
    "- Question: The input question you must answer.\n"
    "- Thought: Your reasoning about the next action.\n"
    "- Action: The JSON object specifying the tool and input.\n"
    "- Observation: The result of the action.\n"
    "- Repeat Thought/Action/Observation as needed.\n"
    "- Conclude with:\n"
    "  Thought: I now know the final answer.\n"
    "  Final Answer: The final answer to the original question.\n\n"
    "Begin your task now. Always use the exact phrase `Final Answer:` when providing a definitive answer."
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
            last_err: Exception
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as err:
                    last_err = err  # type: ignore
                    if attempt == max_retries:
                        logger.error("Retry %d/%d failed: %s", attempt, max_retries, err)
                        raise
                    logger.warning(
                        "Retry %d/%d failed: %r, retrying in %.1fs",
                        attempt,
                        max_retries,
                        err,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            # Should not be reached
            raise last_err  # type: ignore
        return wrapper
    return deco

class OpenAIClient:
    """Thin wrapper around the OpenAI v1 client with retry logic and weather tool."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        """
        Initializes the OpenAI client and sets the system prompt.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set in environment or passed explicitly."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = system_prompt

    @retry(max_retries=3, backoff_factor=2, exceptions=(OpenAIError,))
    def test_connection(self) -> None:
        """Verify that we can list models (i.e., network and auth are OK)."""
        logger.info("Testing OpenAI API connection…")
        self.client.models.list()
        logger.info("Successfully connected to OpenAI API.")

    def chat(
        self,
        question: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send one question to the chat endpoint and return the assistant's reply.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        logger.info("Calling chat.completions.create(model=%r)…", model)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenAIError as e:
            logger.error("Error during chat completion: %s", e)
            raise
        return cast(str, resp.choices[0].message.content).strip()

    def get_weather(self, location: str) -> str:
        """
        Dummy weather function for testing.
        Returns hard-coded weather for Paris, generic otherwise.
        """
        logger.info("Fetching dummy weather for %s…", location)
        if location.lower() == "paris":
            return "Cloudy, 59°F (15°C)"
        return "Sunny, 75°F (24°C)"


def main() -> None:
    """Main entry point for the application."""
    client = OpenAIClient()
    try:
        client.test_connection()
    except OpenAIError as e:
        logger.critical("Connection test failed: %s", e)
        return

    # Chat example
    question = "What's the weather in Paris?"
    try:
        answer = client.chat(question)
        print("Chat says:", answer)
    except OpenAIError as e:
        logger.error("Failed to get chat response: %s", e)

    # Dummy weather test
    print("Dummy weather for Paris:", client.get_weather("Paris"))

if __name__ == "__main__":
    main()
