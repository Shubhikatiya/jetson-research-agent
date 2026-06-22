"""Tests for research_agent.py"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, Mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_agent import (
    ResearchState,
    build_workflow,
    search_node,
    extract_node,
    analyze_node,
    save_node,
    decision_node,
    _strip_html,
)


def test_strip_html():
    html = "<html><body><script>alert(1)</script><p>Hello</p></body></html>"
    result = _strip_html(html)
    assert "Hello" in result
    assert "<script>" not in result
    print("[OK] _strip_html")


@patch("research_agent.DDGS")
async def test_search_node(mock_ddgs_cls):
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=None)
    mock_ddgs.text.return_value = [
        {"title": "T1", "href": "http://a.com", "body": "Snippet 1"},
        {"title": "T2", "href": "http://b.com", "body": "Snippet 2"},
    ]
    mock_ddgs_cls.return_value = mock_ddgs

    state: ResearchState = {
        "topic": "AI chips",
        "sources": [],
        "extracted_content": [],
        "analysis": "",
        "report": "",
        "iterations": 0,
        "local_mode": False,
        "model": "llama3",
        "image_path": None,
    }
    result = await search_node(state)
    assert len(result["sources"]) == 2
    assert result["sources"][0]["title"] == "T1"
    assert result["iterations"] == 1
    print("[OK] search_node")


@patch("research_agent.httpx.AsyncClient")
async def test_extract_node(mock_client_cls):
    mock_resp = Mock()
    mock_resp.text = "<html><body><h1>Title</h1><p>This paragraph has more than fifty characters so it should pass the validation check.</p></body></html>"
    mock_resp.raise_for_status = Mock()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    state: ResearchState = {
        "topic": "AI chips",
        "sources": [{"url": "http://example.com", "content": "snippet"}],
        "extracted_content": [],
        "analysis": "",
        "report": "",
        "iterations": 0,
        "local_mode": False,
        "model": "llama3",
        "image_path": None,
    }
    result = await extract_node(state)
    assert len(result["extracted_content"]) == 1
    print("[OK] extract_node")


@patch("research_agent.get_llm_client")
@patch("research_agent.ChatPromptTemplate")
async def test_analyze_node(mock_prompt_cls, mock_get_llm):
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="Key Facts: AI chips are fast. Summary: Good."))
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)
    mock_prompt_cls.from_messages.return_value = mock_prompt
    mock_get_llm.return_value = MagicMock()

    state: ResearchState = {
        "topic": "AI chips",
        "sources": [],
        "extracted_content": ["content"],
        "analysis": "",
        "report": "",
        "iterations": 0,
        "local_mode": False,
        "model": "llama3",
        "image_path": None,
    }
    result = await analyze_node(state)
    assert "Key Facts" in result["analysis"]
    print("[OK] analyze_node")


async def test_save_node():
    state: ResearchState = {
        "topic": "AI chips",
        "sources": [{"title": "T1", "url": "http://a.com"}],
        "extracted_content": [],
        "analysis": "Summary: AI chips are great.",
        "report": "",
        "iterations": 0,
        "local_mode": False,
        "model": "llama3",
        "image_path": None,
    }
    result = await save_node(state)
    assert result["report"] != ""
    assert os.path.exists("reports")
    print("[OK] save_node")


def test_decision_node_finish():
    state: ResearchState = {
        "topic": "AI chips",
        "sources": [{"title": "T1"}],
        "extracted_content": [],
        "analysis": "Key Facts: xyz. Summary: done.",
        "report": "",
        "iterations": 0,
        "local_mode": False,
        "model": "llama3",
        "image_path": None,
    }
    assert decision_node(state) == "finish"
    print("[OK] decision_node (finish)")


def test_decision_node_continue():
    state: ResearchState = {
        "topic": "AI chips",
        "sources": [{"title": "T1"}],
        "extracted_content": [],
        "analysis": "I need more sources.",
        "report": "",
        "iterations": 1,
        "local_mode": False,
        "model": "llama3",
        "image_path": None,
    }
    assert decision_node(state) == "continue_search"
    print("[OK] decision_node (continue)")


def test_decision_node_no_sources():
    state: ResearchState = {
        "topic": "AI chips",
        "sources": [],
        "extracted_content": [],
        "analysis": "need more sources",
        "report": "",
        "iterations": 2,
        "local_mode": False,
        "model": "llama3",
        "image_path": None,
    }
    assert decision_node(state) == "finish"
    print("[OK] decision_node (no sources = finish)")


@patch("research_agent.DDGS")
@patch("research_agent.get_llm_client")
@patch("research_agent.ChatPromptTemplate")
async def test_full_workflow(mock_prompt_cls, mock_get_llm, mock_ddgs_cls):
    # Mock DDGS search
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=None)
    mock_ddgs.text.return_value = [
        {"title": "T1", "href": "http://a.com", "body": "C1"},
    ]
    mock_ddgs_cls.return_value = mock_ddgs

    # Mock httpx for page fetch
    fetch_resp = Mock()
    fetch_resp.text = "<html><body><h1>Test</h1><p>AI chips are fast and powerful with more than fifty characters.</p></body></html>"
    fetch_resp.raise_for_status = Mock()

    with patch("research_agent.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fetch_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="Key Facts: fast. Summary: good."))
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_cls.from_messages.return_value = mock_prompt
        mock_get_llm.return_value = MagicMock()

        workflow = build_workflow()
        app = workflow.compile()

        state: ResearchState = {
            "topic": "AI chips",
            "sources": [],
            "extracted_content": [],
            "analysis": "",
            "report": "",
            "iterations": 0,
            "local_mode": True,
            "model": "llama3",
            "image_path": None,
        }

        result = await app.ainvoke(state)
        assert result["report"] != ""
        assert "AI chips" in result["report"]
        print("[OK] full workflow")


async def main():
    test_strip_html()
    await test_search_node()
    await test_extract_node()
    await test_analyze_node()
    await test_save_node()
    test_decision_node_finish()
    test_decision_node_continue()
    test_decision_node_no_sources()
    await test_full_workflow()
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
