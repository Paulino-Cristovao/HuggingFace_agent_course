#!/usr/bin/env python3

"""
DuckDuckGo-powered agent using a local Ollama LLM.

This script initializes a ReAct-style agent that uses a local Ollama LLM
and DuckDuckGo as a search tool. The agent can answer user queries by
leveraging the LLM and performing web searches when necessary.
"""

from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from rich.console import Console
from rich.panel import Panel


def configure_llm() -> Ollama:
    """
    Configure and return the local Ollama LLM.

    Returns:
        Ollama: An instance of the Ollama LLM with specified parameters.
    """
    llm = Ollama(model="llama3.2:latest")
    llm.temperature = 0.2
    llm.num_ctx = 4096
    return llm


def initialize_search_agent(llm: Ollama) -> initialize_agent:
    """
    Initialize and return a ReAct-style agent with DuckDuckGo search tool.

    Args:
        llm (Ollama): The local Ollama LLM instance.

    Returns:
        initialize_agent: The initialized agent with search capabilities.
    """
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )


def main() -> None:
    """
    Main function to run the agent.

    This function sets up the console interface, initializes the LLM and agent,
    and enters a loop to process user queries until interrupted.
    """
    console = Console()
    console.print("\n[bold]Ask me anything! I’ll search DuckDuckGo when needed.[/bold]\n")

    llm = configure_llm()
    agent = initialize_search_agent(llm)

    while True:
        try:
            user_query = console.input("[bold green]› [/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye! 👋")
            break

        if not user_query.strip():
            continue

        # Run the agent and display the response
        answer = agent.run(user_query)
        console.print(
            Panel.fit(
                answer,
                title="🤖  Ollama Agent",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    main()
