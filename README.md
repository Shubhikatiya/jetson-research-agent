# Research Agent

An **async multi-step research agent** built with **LangGraph** and **LangChain** that queries the web via DuckDuckGo, fetches and extracts source content, synthesizes findings with an LLM, and generates a Markdown report.

Supports both **local Ollama** (edge deployment) and **DeepSeek API** (cloud) backends. Includes an optional **Gradio web UI** for interactive research with image upload support.

---

## Features

- **LangGraph workflow** — stateful multi-step pipeline: search → extract → analyze → save
- **DuckDuckGo search** with caching and retry logic
- **HTML extraction** with BeautifulSoup + httpx fallback strategies
- **LLM synthesis** via LangChain OpenAI-compatible interface
- **Vision support** — analyze images to generate research topics
- **Dual backend** — local Ollama (edge) or DeepSeek API (cloud)
- **Gradio web UI** — chat interface with real-time status updates
- **Iterative research** — auto-loops when analysis signals "need more sources"

---

## Architecture

```
User Query / Image
    |
    v
[start_node] --(has image?)--> [vision_node] --> [search_node]
    |                                    |
    +--(text only)-----------------------+
    |
    v
[extract_node] --> [analyze_node]
    |
    v
[decision_node] --(need more sources?)--> [search_node]
    |
    +--(done)--------------------------> [save_node] --> Markdown report
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Ollama or DeepSeek credentials
```

### 3. Start Ollama (for local mode)

```bash
ollama serve
# Ensure your model is pulled: ollama pull qwen2.5:3b-fast
```

---

## CLI Usage

```bash
# Local mode (Ollama)
python research_agent.py "latest AI chip benchmarks 2026" --local

# With specific model
python research_agent.py "quantum computing breakthroughs" --local --model llama3

# Image-based research
python research_agent.py --image screenshot.png --local

# Cloud mode (DeepSeek API)
export DEEPSEEK_API_KEY=your_key
python research_agent.py "space exploration 2026"
```

---

## Web UI

```bash
# Local mode UI
python ui.py --local

# With public share link
python ui.py --local --share

# Custom port
python ui.py --local --port 7860
```

Open browser at `http://localhost:7860`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_GENERATE_URL` | `http://localhost:11434/api/generate` | Ollama native generate endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b-fast` | Default local model |
| `VISION_MODEL` | `qwen2.5:3b-fast` | Model for image analysis |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (cloud mode) |

---

## Testing

```bash
python test_research_agent.py
```

Tests cover:
- HTML stripping (`BeautifulSoup`)
- Search node (mocked DuckDuckGo)
- Extract node (mocked httpx)
- Analyze node (mocked LLM chain)
- Save node (file I/O)
- Decision routing logic
- Full LangGraph workflow integration

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph + LangChain |
| LLM Client | langchain-openai (OpenAI-compatible) |
| Search | DuckDuckGo (`ddgs`) |
| HTTP Client | httpx (async) |
| HTML Parsing | BeautifulSoup4 |
| Web UI | Gradio |
| State Management | TypedDict + LangGraph StateGraph |

---

## Workflow Nodes

| Node | Purpose |
|------|---------|
| `start_node` | Routes to vision or search based on input |
| `vision_node` | Analyzes uploaded image, sets topic from description |
| `search_node` | Queries DuckDuckGo with caching and retry logic |
| `extract_node` | Fetches page text with User-Agent fallback |
| `analyze_node` | Synthesizes sources into key facts, contradictions, gaps, summary |
| `decision_node` | Routes to continue search or finish based on analysis |
| `save_node` | Writes Markdown report to `reports/` directory |

---

## License

MIT
