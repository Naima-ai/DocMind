# DocMind: Agentic Document Intelligence Platform

A Streamlit application that combines Retrieval-Augmented Generation (RAG)
with predictive analytics inside a single **tool-using agent**, rather than
a single hardcoded prompt. Claude decides, per user message, whether it
needs to search the ingested documents, run a numeric forecast, or check
for risk/compliance language — and can chain several of these calls
together before answering.

## What it does

1. **Document ingestion & semantic search** — Upload PDF/DOCX/TXT files.
   Text is chunked, embedded locally with `sentence-transformers`
   (`all-MiniLM-L6-v2`), and indexed with FAISS for fast similarity search.
2. **Grounded Q&A** — The agent retrieves the most relevant chunks for a
   question before answering, and cites which source file it used.
3. **Predictive analytics** — Upload a CSV/XLSX of structured data (e.g.
   monthly revenue, ticket counts, defect rates). The agent can forecast
   future values using linear trend regression, both through the chat
   interface and from the manual dashboard, with a Plotly chart.
4. **Risk assessment** — Scans ingested documents for compliance/risk
   language (breach, penalty, liability, termination, etc.) and returns a
   scored risk level.
5. **Real agent loop** — Tools are defined with JSON schemas and passed to
   Claude via the Anthropic tool-use API. The app runs the standard
   agent loop: send message → Claude requests a tool → app executes the
   Python function → result is returned to Claude → Claude either calls
   another tool or gives a final answer.


```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste an Anthropic API key into the sidebar (get one at
console.anthropic.com). No key is required to upload/preview documents or
datasets — only for the chat agent and its tool calls.

## Notes / possible extensions

- Swap `IndexFlatL2` for `IndexIVFFlat` or a persistent vector DB (Chroma,
  Pinecone) if you need it to scale beyond a single session.
- Swap the linear-regression forecaster for `statsmodels` exponential
  smoothing or Prophet if you want seasonality-aware forecasts.
- Add a `send_slack_alert` tool to make the risk assessment actionable.
- Persist `st.session_state.chunks` / dataframes to disk or a database so
  ingested knowledge survives a server restart.

