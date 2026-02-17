# Healthcare Research Agent

A2A healthcare research agent using Google ADK and Google Search.

## Setup

1. Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the agent:
```bash
uv run python research_agent/research_agent.py
```

The agent will be available at `http://localhost:8001`