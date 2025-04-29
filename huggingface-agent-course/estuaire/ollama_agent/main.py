#!/usr/bin/env python3
"""
Minimal DuckDuckGo‑powered chat agent that runs on a local Ollama model.

Updates in this version
-----------------------
* **More head‑room** – raises LangChain’s default guard‑rails to
  `max_iterations=25` and removes the 60‑second overall limit.
* **English‑only action blocks** – adds a one‑sentence hint so the LLM keeps
  `Action:` / `Action Input:` lines in English even if the user asks for another
  language, avoiding the "Agent stopped due to iteration limit" issue.

Language selection is still entirely prompt‑based: just ask in Spanish or add
“Please answer in Spanish.”

Environment variables
---------------------
MODEL   – Ollama model name   (default: ``deepseek-r1:1.5b``)
TEMP    – generation temperature (default: ``0.2``)
CTX     – context length (default: ``4096``)
MAX_ITERS – iteration cap before early‑stop (default: ``25``)
"""
from __future__ import annotations

import os

from langchain.agents import AgentType, initialize_agent
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from rich.console import Console
from rich.panel import Panel

# ── Configuration ────────────────────────────────────────────────────────────
MODEL = os.getenv("MODEL", "mistral")
TEMPERATURE = float(os.getenv("TEMP", "0.2"))
CONTEXT_LEN = int(os.getenv("CTX", "4096"))
MAX_ITERS = int(os.getenv("MAX_ITERS", "25"))

# ── Helpers ─────────────────────────────────────────────────────────────────

def make_llm() -> Ollama:
    """Return a configured Ollama wrapper."""
    return Ollama(model=MODEL, temperature=TEMPERATURE, num_ctx=CONTEXT_LEN)


def make_agent():
    """Build a ReAct‑style agent that can use DuckDuckGo."""
    tools = [DuckDuckGoSearchRun()]
    return initialize_agent(
        tools=tools,
        llm=make_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=MAX_ITERS,
        max_execution_time=None,  # unlimited wall‑clock time
        early_stopping_method="generate",
    )


# ── Chat loop ────────────────────────────────────────────────────────────────

TOOL_CALL_HINT = (
    "When you need to call a tool, ALWAYS write the action block in English\n"
    "(e.g. Action: DuckDuckGoSearchRun). Otherwise, follow the user's language.\n\n"
)

def chat_loop() -> None:
    console = Console()
    console.print(
        "\n[bold cyan]Ask me anything! I’ll use DuckDuckGo when I need to look things up.\n"
        "If you want the reply in a specific language, just say so in your prompt.[/bold cyan]\n"
    )

    agent = make_agent()

    while True:
        try:
            user_query = console.input("[bold green]› [/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[italic]Bye! 👋[/]")
            break

        if not user_query.strip():
            continue

        # Pre‑pend the English‑only tool‑call hint
        query = TOOL_CALL_HINT + user_query

        answer = agent.run(query)
        console.print(
            Panel.fit(
                answer,
                title="🤖  Ollama Agent",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            )
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        chat_loop()
    except Exception as exc:  # noqa: BLE001
        Console().print(f"[red]Unhandled error:[/] {exc}")
