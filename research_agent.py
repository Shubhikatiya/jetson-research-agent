"""
Async Multi-Step Research Agent using LangGraph and LangChain.

Queries the web with DuckDuckGo search, fetches sources, synthesizes
findings with an LLM, and saves the report.

Usage:
    python research_agent.py "latest AI chip benchmarks 2026" --local
"""

import argparse
import asyncio
import base64
import os
import re
import sys
from typing import Dict, List, Optional, TypedDict, Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph


class ResearchState(TypedDict):
    topic: str
    sources: List[Dict[str, Any]]
    extracted_content: List[str]
    analysis: str
    report: str
    iterations: int
    local_mode: bool
    model: str
    image_path: Optional[str]


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_GENERATE_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
MAX_ITERATIONS = 3
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-fast")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2.5:3b-fast")
SEARCH_DELAY_SECONDS = 5  # Delay between searches to avoid rate limits
SEARCH_CACHE: Dict[str, List[Dict[str, Any]]] = {}  # Simple in-memory cache


def get_llm_client(state: ResearchState):
    """Return an async LangChain-compatible chat model."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("pip install langchain-openai")

    model = state.get("model", DEFAULT_OLLAMA_MODEL)

    if not state["local_mode"]:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            return ChatOpenAI(
                model="deepseek-chat",
                api_key=deepseek_key,
                base_url="https://api.deepseek.com/v1",
                temperature=0.2,
            )

    return ChatOpenAI(
        model=model,
        api_key="ollama",
        base_url=OLLAMA_URL,
        temperature=0.2,
    )


# ────────────────────────── SEARCH ──────────────────────────

async def search_node(state: ResearchState) -> ResearchState:
    """Search DuckDuckGo with caching and rate limiting."""
    topic = state["topic"]
    model = state.get("model", DEFAULT_OLLAMA_MODEL)
    print(f"[search] querying for: {topic}")

    # Check cache first
    cache_key = topic.lower().strip()
    if cache_key in SEARCH_CACHE:
        print(f"[search] returning cached results for: {topic}")
        state["sources"] = SEARCH_CACHE[cache_key]
        state["iterations"] = state.get("iterations", 0) + 1
        return state

    results: List[Dict[str, Any]] = []

    # Try DuckDuckGo with retries and delays
    for attempt in range(3):
        if results:
            break
        try:
            print(f"[search] DuckDuckGo attempt {attempt + 1}/3")
            with DDGS() as ddgs:
                for r in ddgs.text(topic, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", ""),
                    })
            print(f"[search] DuckDuckGo found {len(results)} results")
        except Exception as exc:
            print(f"[search] DuckDuckGo attempt {attempt + 1} failed: {exc}")
            if attempt < 2:
                wait_time = SEARCH_DELAY_SECONDS * (attempt + 1)
                print(f"[search] waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

    # Fallback: Ollama knowledge
    if not results:
        print("[search] no web results, using Ollama knowledge fallback")
        try:
            results = await _ollama_knowledge_fallback(topic, model)
        except Exception as exc:
            print(f"[search] Ollama fallback failed: {exc}")

    # Cache results
    if results:
        SEARCH_CACHE[cache_key] = results

    print(f"[search] total sources: {len(results)}")
    state["sources"] = results
    state["iterations"] = state.get("iterations", 0) + 1
    return state


async def _ollama_knowledge_fallback(topic: str, model: str) -> List[Dict[str, Any]]:
    """Ask Ollama to answer from its knowledge when web search fails.
    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    """
    print(f"[search/ollama-fallback] asking {model}")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a knowledgeable research assistant. Provide factual, detailed answers."},
            {"role": "user", "content": f"Provide detailed factual information about: {topic}\n\nInclude key facts, specifics, and technical details."},
        ],
        "stream": False,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    if answer:
        print(f"[search/ollama-fallback] generated {len(answer)} chars")
        return [{
            "title": f"Knowledge: {topic}",
            "url": "internal://ollama",
            "content": answer,
        }]
    return []


# ────────────────────────── VISION ──────────────────────────

async def vision_node(state: ResearchState) -> ResearchState:
    """Read image with Ollama vision and set topic."""
    image_path = state.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return state

    print(f"[vision] analyzing image: {image_path}")
    with open(image_path, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("utf-8")

    user_prompt = state["topic"] or "Describe this image in detail."
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            description = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if description:
                state["topic"] = description
                print(f"[vision] topic: {description[:80]}...")
            else:
                print("[vision] no description returned")
        except Exception as exc:
            print(f"[vision] failed: {exc}")

    return state


def start_node(state: ResearchState) -> str:
    """Route to vision if image present."""
    image_path = state.get("image_path")
    if image_path and os.path.exists(image_path):
        return "vision"
    return "search"


# ────────────────────────── EXTRACT ──────────────────────────

async def extract_node(state: ResearchState) -> ResearchState:
    """Fetch page text. Always includes search result metadata as fallback."""
    sources = state["sources"]
    contents: List[str] = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for src in sources:
            url = src.get("url", "")
            title = src.get("title", "")
            snippet = src.get("content", "")

            if not url or url.startswith("internal://"):
                contents.append(snippet)
                continue
            if not url:
                continue

            print(f"[extract] fetching: {url}")
            page_text = ""

            # Try direct fetch
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                page_text = _strip_html(resp.text)
                if len(page_text.strip()) >= 50:
                    contents.append(page_text)
                    print(f"[extract] success ({len(page_text)} chars)")
                    continue
            except Exception:
                pass

            # Retry with User-Agent
            if not page_text or len(page_text.strip()) < 50:
                try:
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    resp = await client.get(url, follow_redirects=True, headers=headers, timeout=20.0)
                    resp.raise_for_status()
                    page_text = _strip_html(resp.text)
                    if len(page_text.strip()) >= 50:
                        contents.append(page_text)
                        print(f"[extract] success UA ({len(page_text)} chars)")
                        continue
                except Exception:
                    pass

            # Always include search result metadata as content
            # Even if webpage fetch fails, the search result itself has useful info
            fallback_parts = [f"Source: {url}"]
            if title:
                fallback_parts.append(f"Title: {title}")
            if snippet:
                fallback_parts.append(f"Snippet: {snippet}")
            if page_text and len(page_text.strip()) > 0:
                fallback_parts.append(f"Extracted text: {page_text[:500]}")

            fallback = "\n".join(fallback_parts)
            contents.append(fallback)
            print(f"[extract] using metadata ({len(fallback)} chars)")

    state["extracted_content"] = contents
    return state


def _strip_html(html: str) -> str:
    """Remove HTML tags."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


# ────────────────────────── ANALYZE ──────────────────────────

async def analyze_node(state: ResearchState) -> ResearchState:
    """Synthesize findings with LLM."""
    contents = state["extracted_content"]
    topic = state["topic"]

    if not contents:
        state["analysis"] = "No sources available. need more sources"
        return state

    joined = "\n\n---SOURCE---\n\n".join(contents)
    prompt_text = (
        f"You are a research assistant. The topic is: {topic}\n\n"
        "Below are sources of information. Analyze them and provide:\n"
        "1. Key Facts\n"
        "2. Contradictions\n"
        "3. Gaps\n"
        "4. Summary\n\n"
        f"Sources:\n{joined}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a precise research assistant. Analyze the sources below thoroughly."),
        HumanMessage(content=prompt_text),
    ])

    llm = get_llm_client(state)
    chain = prompt | llm

    print("[analyze] invoking LLM...")
    try:
        response = await chain.ainvoke({})
        analysis = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        print(f"[analyze] LLM failed: {exc}")
        analysis = f"Error: {exc}"

    state["analysis"] = analysis
    print("[analyze] done")
    return state


# ────────────────────────── DECISION ──────────────────────────

def decision_node(state: ResearchState) -> str:
    """Route to continue or finish."""
    analysis = state.get("analysis", "").lower()
    iterations = state.get("iterations", 0)
    sources = state.get("sources", [])

    if iterations >= MAX_ITERATIONS:
        return "finish"

    if not sources:
        return "finish"

    if "need more sources" in analysis:
        return "continue_search"

    return "finish"


# ────────────────────────── SAVE ──────────────────────────

async def save_node(state: ResearchState) -> ResearchState:
    """Save report to file."""
    topic = state["topic"]
    analysis = state["analysis"]
    sources = state["sources"]
    lines = [f"# Research Report: {topic}\n", "## Analysis\n", analysis, "\n## Sources\n"]
    for idx, src in enumerate(sources, start=1):
        lines.append(f"{idx}. [{src.get('title', 'N/A')}]({src.get('url', 'N/A')})")

    report = "\n".join(lines)
    state["report"] = report

    safe = re.sub(r"[^\w\-]", "_", topic)
    filename = f"{safe}.md"
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[save] saved to: {path}")
    return state


# ────────────────────────── WORKFLOW ──────────────────────────

def build_workflow() -> StateGraph:
    workflow = StateGraph(ResearchState)
    workflow.add_node("vision", vision_node)
    workflow.add_node("search", search_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("save", save_node)

    workflow.set_conditional_entry_point(
        start_node, {"vision": "vision", "search": "search"}
    )
    workflow.add_edge("vision", "search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "analyze")
    workflow.add_conditional_edges(
        "analyze", decision_node, {"continue_search": "search", "finish": "save"}
    )
    workflow.add_edge("save", END)
    return workflow


# ────────────────────────── ENTRYPOINT ──────────────────────────

async def run_research(
    topic: str,
    local_mode: bool = False,
    model: str = "llama3",
    image_path: Optional[str] = None,
) -> None:
    workflow = build_workflow()
    app = workflow.compile()

    state: ResearchState = {
        "topic": topic, "sources": [], "extracted_content": [],
        "analysis": "", "report": "", "iterations": 0,
        "local_mode": local_mode, "model": model, "image_path": image_path,
    }

    print(f"Starting: {topic}")
    result = await app.ainvoke(state)
    print("\n=== Report ===\n")
    print(result.get("report", "No report."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="?", default="")
    parser.add_argument("--image")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3"))
    args = parser.parse_args()

    topic = args.topic
    if args.image:
        if not topic:
            topic = "image research"
    elif not topic:
        parser.error("Provide --topic or --image")

    asyncio.run(run_research(topic, local_mode=args.local, model=args.model, image_path=args.image))


if __name__ == "__main__":
    main()
