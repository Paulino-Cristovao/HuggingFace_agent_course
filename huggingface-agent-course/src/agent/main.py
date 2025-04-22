import os
import time
import logging
from typing import Any, Callable, Optional, TypeVar, cast
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
SYSTEM_PROMPT = """You are an AI assistant that answers questions to the best of your ability. You can use tools to gather information when needed. Always provide clear, concise, and accurate answers.

When using tools, follow this process:
1. Think about the action you need to take.
2. Use the tool by specifying a JSON object with the following structure:
    {
      "action": "<tool_name>",
      "action_input": {<input_parameters>}
    }
3. Observe the result of the action and decide the next step.

Example tool usage:
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}

Always follow this format:
- Question: The input question you must answer.
- Thought: Your reasoning about the next action.
- Action: The JSON object specifying the tool and input.
- Observation: The result of the action.
- Repeat Thought/Action/Observation as needed.
- Conclude with:
  Thought: I now know the final answer.
  Final Answer: The final answer to the original question.

Begin your task now. Always use the exact phrase `Final Answer:` when providing a definitive answer."""

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
                        logger.error(f"Retry {attempt}/{max_retries} failed: {err}")
                        raise
                    logger.warning(
                        f"Retry {attempt}/{max_retries} failed: {err!r}, retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            raise last_err  # type: ignore
        return wrapper
    return deco

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, system_prompt: str = SYSTEM_PROMPT) -> None:
        """
        Wraps the v1+ OpenAI client, optionally injecting a custom system prompt.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed explicitly.")
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = system_prompt

    @retry(max_retries=3, backoff_factor=2, exceptions=(OpenAIError,))
    def test_connection(self) -> None:
        """List models to verify network/auth is OK."""
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
        Sends a single-turn chat completion, automatically prepending the system_prompt.
        The user only needs to pass the raw question.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        logger.info(f"Calling chat.completions.create(model={model!r})…")
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return cast(str, resp.choices[0].message.content).strip()

    def get_weather(self, location: str) -> str:
        """
        Dummy weather function for testing.
        Returns a hard‑coded weather string for any input.
        """
        logger.info(f"Fetching dummy weather for {location}…")
        # You could later replace this with a real API call
        if location.lower() == "paris":
            return "Cloudy, 59°F (15°C)"
        return "Sunny, 75°F (24°C)"


def main() -> None:
    """Main entry point for the application."""
    client = OpenAIClient()
    try:
        client.test_connection()
    except Exception as e:
        logger.critical(f"Connection test failed: {e}")
        return

    # Test chat
    question = "What's the weather in Paris?"
    try:
        answer = client.chat(question)
        print("Chat says:", answer)
    except Exception as e:
        logger.error(f"Failed to get chat response: {e}")

    # Test dummy weather
    print("Dummy get_weather →", client.get_weather("Paris"))


if __name__ == "__main__":
    main()
