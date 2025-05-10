"""
Example of using OpenRouter as an LLM provider with Browser Use.

@dev You need to add OPENROUTER_API_KEY to your environment variables.
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv('OPENROUTER_API_KEY', '')
if not api_key:
    raise ValueError('OPENROUTER_API_KEY is not set')

# Optional: Set site information for OpenRouter analytics
site_url = os.getenv('SITE_URL', '')
site_name = os.getenv('SITE_NAME', 'Browser Use Example')


async def run_search(model_name: str = "openai/gpt-4o", use_vision: bool = True):
    """
    Run a browser automation task using an OpenRouter model.
    
    Args:
        model_name: The OpenRouter model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet")
        use_vision: Whether to enable vision capabilities
    """
    # Initialize the LLM with OpenRouter
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr(api_key),
        model=model_name,
        model_kwargs={
            "headers": {
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            }
        },
    )
    
    # Create the agent with the OpenRouter LLM
    agent = Agent(
        task=(
            'Go to amazon.com, search for "wireless headphones", '
            'filter by "Highest customer rating", and '
            'return the title and price of the first product'
        ),
        llm=llm,
        use_vision=use_vision,
    )

    # Run the agent
    result = await agent.run()
    return result


async def main():
    """Run examples with different OpenRouter models"""
    
    # Example 1: Using OpenAI's GPT-4o through OpenRouter
    print("\n=== Running with OpenAI GPT-4o via OpenRouter ===")
    await run_search(model_name="openai/gpt-4o")
    
    # Example 2: Using Anthropic's Claude model through OpenRouter
    print("\n=== Running with Anthropic Claude via OpenRouter ===")
    await run_search(model_name="anthropic/claude-3-5-sonnet", use_vision=False)
    
    # Example 3: Using a different model (customize as needed)
    # print("\n=== Running with Custom Model via OpenRouter ===")
    # await run_search(model_name="your/preferred-model", use_vision=False)


if __name__ == '__main__':
    asyncio.run(main())
