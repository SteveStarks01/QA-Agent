"""
Test OpenRouter integration with Browser Use.

This test verifies that the OpenRouter integration works correctly.
It requires an OPENROUTER_API_KEY environment variable to be set.
"""

import asyncio
import os
import sys

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser

# Load environment variables
load_dotenv()

# Skip tests if OPENROUTER_API_KEY is not set
api_key = os.getenv('OPENROUTER_API_KEY', '')
if not api_key:
    pytest.skip("OPENROUTER_API_KEY not set", allow_module_level=True)


@pytest.fixture
def llm():
    """Initialize OpenRouter LLM for testing"""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr(api_key),
        model="openai/gpt-4o",
        model_kwargs={
            "headers": {
                "HTTP-Referer": "https://github.com/browser-use/browser-use",
                "X-Title": "Browser Use Test",
            }
        },
    )


@pytest.fixture(scope='session')
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def browser():
    """Initialize browser for testing"""
    browser = Browser()
    yield browser
    await browser.close()


@pytest.fixture
async def context(browser):
    """Initialize browser context for testing"""
    async with await browser.new_context() as context:
        yield context


@pytest.mark.asyncio
async def test_openrouter_search(llm, context):
    """Test basic search functionality with OpenRouter"""
    agent = Agent(
        task="Search Google for 'Browser Use'",
        llm=llm,
        browser_context=context,
    )
    
    history: AgentHistoryList = await agent.run(max_steps=3)
    
    # Verify that the agent performed a search action
    action_names = history.action_names()
    assert any(action in action_names for action in ['search_google', 'go_to_url'])
    
    # Verify that the agent didn't encounter errors
    assert not history.has_errors(), f"Errors encountered: {history.errors()}"


@pytest.mark.asyncio
async def test_openrouter_navigation(llm, context):
    """Test navigation functionality with OpenRouter"""
    agent = Agent(
        task="Go to example.com and return the title of the page",
        llm=llm,
        browser_context=context,
    )
    
    history: AgentHistoryList = await agent.run(max_steps=3)
    
    # Verify that the agent navigated to a URL
    urls = history.urls()
    assert any(url and 'example.com' in url for url in urls), "Failed to navigate to example.com"
    
    # Verify that the agent extracted content (page title)
    content = history.extracted_content()
    assert any(content), "No content was extracted"


if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-xvs", __file__])
