"""
DocMind — Agentic Document Intelligence Platform
--------------------------------------------------
A multi-purpose AI agent that:
  1. Ingests company documents (PDF / DOCX / TXT) and builds a semantic
     search index (RAG) so it can answer questions grounded in real content.
  2. Ingests structured data (CSV / XLSX) and can forecast numeric trends
     (e.g. revenue, complaints, defect counts) using regression.
  3. Scans ingested documents for risk / compliance language and produces
     a risk score.
  4. Wires all of the above together as *tools* that Claude decides to call
     on its own (Anthropic tool-use / function-calling), so this is a real
     agent loop, not a single hardcoded prompt template.

Run with:  streamlit run app.py
"""

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from pypdf import PdfReader
import docx
import anthropic

# --------------------------------------------------------------------------
# Page / session setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="DocMind — Agentic Doc AI", page_icon="🧠", layout="wide")

MODEL_OPTIONS = ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"]

if "chunks" not in st.session_state:
    st.session_state.chunks = []          # list[str]  text chunks from all docs
    st.session_state.chunk_sources = []   # list[str]  which file each chunk came from
    st.session_state.index = None         # faiss index
    st.session_state.dataframes = {}      # name -> pd.DataFrame
    st.session_state.messages = []        # chat history (Claude message format)
    st.session_state.display_messages = []  # chat history (for rendering)
    st.session_state.last_forecast = None


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------------------------------
# Document ingestion
# --------------------------------------------------------------------------
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        d = docx.Document(uploaded_file)
        return "\n".join(p.text for p in d.paragraphs)
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def add_document_to_index(uploaded_file):
    embedder = load_embedder()
    text = extract_text(uploaded_file)
    if not text.strip():
        return 0
    new_chunks = chunk_text(text)
    embeddings = embedder.encode(new_chunks, show_progress_bar=False).astype("float32")

    if st.session_state.index is None:
        dim = embeddings.shape[1]
        st.session_state.index = faiss.IndexFlatL2(dim)

    st.session_state.index.add(embeddings)
    st.session_state.chunks.extend(new_chunks)
    st.session_state.chunk_sources.extend([uploaded_file.name] * len(new_chunks))
    return len(new_chunks)


# --------------------------------------------------------------------------
# Agent tools — plain Python functions the model can invoke
# --------------------------------------------------------------------------
def search_documents(query: str, k: int = 4) -> str:
    if st.session_state.index is None or not st.session_state.chunks:
        return "No documents have been ingested yet."
    embedder = load_embedder()
    q_emb = embedder.encode([query]).astype("float32")
    k = min(k, len(st.session_state.chunks))
    _, ids = st.session_state.index.search(q_emb, k)
    results = []
    for i in ids[0]:
        if 0 <= i < len(st.session_state.chunks):
            src = st.session_state.chunk_sources[i]
            results.append(f"[source: {src}]\n{st.session_state.chunks[i]}")
    return "\n\n---\n\n".join(results) if results else "No relevant passages found."


def forecast_column(dataset_name: str, column_name: str, periods_ahead: int = 3) -> str:
    df = st.session_state.dataframes.get(dataset_name)
    if df is None:
        available = list(st.session_state.dataframes.keys())
        return f"Dataset '{dataset_name}' not found. Available datasets: {available}"
    if column_name not in df.columns:
        return f"Column '{column_name}' not found. Available columns: {list(df.columns)}"

    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if len(series) < 3:
        return "Not enough numeric data points in that column to forecast (need at least 3)."

    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(series), len(series) + periods_ahead).reshape(-1, 1)
    preds = model.predict(future_X)
    slope = float(model.coef_[0])
    trend = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"

    st.session_state.last_forecast = {
        "dataset": dataset_name,
        "column": column_name,
        "history": series.tolist(),
        "forecast": preds.tolist(),
        "trend": trend,
        "slope": slope,
    }
    return (
        f"Trend for '{column_name}' in dataset '{dataset_name}': {trend} "
        f"(slope={slope:.3f} per row). "
        f"Forecast for the next {periods_ahead} period(s): {[round(p, 2) for p in preds]}"
    )


RISK_KEYWORDS = [
    "penalty", "breach", "terminate", "termination", "liability", "lawsuit",
    "default", "non-compliance", "noncompliance", "violation", "fine",
    "overdue", "dispute", "late payment", "indemnify", "audit finding",
]


def assess_risk() -> str:
    if not st.session_state.chunks:
        return "No documents ingested yet."
    full_text = " ".join(st.session_state.chunks).lower()
    counts = {kw: full_text.count(kw) for kw in RISK_KEYWORDS if full_text.count(kw) > 0}
    score = sum(counts.values())
    level = "Low" if score < 3 else "Medium" if score < 8 else "High"
    detail = ", ".join(f"{k} ({v})" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    detail = detail or "no risk-related terms detected"
    return f"Risk level: {level} (score={score}). Flagged terms: {detail}"


TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Semantically search the ingested documents (PDF/DOCX/TXT) for passages "
            "relevant to a question. Always use this before answering any factual "
            "question about document content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "forecast_column",
        "description": (
            "Forecast future values of a numeric column from an uploaded structured "
            "dataset (CSV/Excel) using linear trend regression. Use this whenever the "
            "user asks about future values, projections, or trends in numeric data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Name of the uploaded dataset"},
                "column_name": {"type": "string", "description": "Numeric column to forecast"},
                "periods_ahead": {
                    "type": "integer",
                    "description": "How many future periods to predict",
                    "default": 3,
                },
            },
            "required": ["dataset_name", "column_name"],
        },
    },
    {
        "name": "assess_risk",
        "description": (
            "Scan all ingested documents for risk/compliance-related keywords "
            "(penalties, breaches, liabilities, disputes, etc.) and return a risk "
            "score and level. Use when the user asks about risk, compliance, or concerns."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_FUNCS = {
    "search_documents": lambda inp: search_documents(inp["query"]),
    "forecast_column": lambda inp: forecast_column(
        inp["dataset_name"], inp["column_name"], inp.get("periods_ahead", 3)
    ),
    "assess_risk": lambda inp: assess_risk(),
}

SYSTEM_PROMPT = (
    "You are DocMind, an enterprise AI agent. You help employees find answers inside "
    "ingested company documents and make data-driven predictions from structured "
    "datasets. Always call search_documents before answering factual questions about "
    "document content — never invent content. Call forecast_column when the user asks "
    "about trends, projections, or future values in a dataset. Call assess_risk when "
    "the user asks about risk, compliance, or concerns. State clearly which tool's "
    "findings your answer relies on. Be concise, precise, and say when the documents "
    "don't contain enough information to answer."
)


def run_agent(client: anthropic.Anthropic, model: str, user_message: str) -> str:
    st.session_state.messages.append({"role": "user", "content": user_message})
    messages = st.session_state.messages

    for _ in range(6):  # cap tool-use loop iterations for safety
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Use the SDK's own serialization so every block type (text,
            # tool_use, thinking, etc.) round-trips correctly — a hand-rolled
            # serializer that only knows about "text"/"tool_use" will silently
            # drop other block types and break the next API call.
            assistant_content = [b.model_dump() for b in response.content]
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = TOOL_FUNCS[block.name](block.input)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
                    )
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
            continue

        final_text = "".join(b.text for b in response.content if b.type == "text")
        messages.append({"role": "assistant", "content": final_text})
        return final_text

    return "I hit the tool-use step limit without reaching a final answer — try rephrasing."


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 DocMind")
    st.caption("Agentic document QA + predictive analytics")
    api_key = st.text_input("Anthropic API key", type="password")
    model = st.selectbox("Model", MODEL_OPTIONS, index=0)
    st.divider()
    st.markdown("**Ingested documents:** " + str(len(set(st.session_state.chunk_sources))))
    st.markdown("**Chunks indexed:** " + str(len(st.session_state.chunks)))
    st.markdown("**Datasets loaded:** " + str(len(st.session_state.dataframes)))
    if st.button("Reset session"):
        for key in ["chunks", "chunk_sources", "index", "dataframes",
                    "messages", "display_messages", "last_forecast"]:
            st.session_state.pop(key, None)
        st.rerun()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_ingest, tab_chat, tab_predict = st.tabs(
    ["📁 Ingest", "💬 Chat Agent", "📊 Predictions Dashboard"]
)

# ---- Ingest tab ----
with tab_ingest:
    st.subheader("Upload documents for Q&A")
    doc_files = st.file_uploader(
        "PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    if doc_files and st.button("Ingest documents"):
        with st.spinner("Chunking and embedding..."):
            total = 0
            for f in doc_files:
                total += add_document_to_index(f)
        st.success(f"Ingested {len(doc_files)} file(s) into {total} searchable chunks.")

    st.divider()
    st.subheader("Upload structured data for predictions")
    data_files = st.file_uploader(
        "CSV or Excel", type=["csv", "xlsx"], accept_multiple_files=True, key="data_upload"
    )
    if data_files and st.button("Load datasets"):
        for f in data_files:
            df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
            st.session_state.dataframes[f.name] = df
        st.success(f"Loaded {len(data_files)} dataset(s).")

    if st.session_state.dataframes:
        st.subheader("Loaded datasets preview")
        for name, df in st.session_state.dataframes.items():
            with st.expander(name):
                st.dataframe(df.head(20))

# ---- Chat tab ----
with tab_chat:
    st.subheader("Ask DocMind")
    st.caption(
        "Ask questions about your documents, request a forecast on loaded data, "
        "or ask for a risk assessment. The agent decides which tool(s) to use."
    )

    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. 'Summarize the payment terms' or 'Forecast next quarter revenue'")
    if prompt:
        if not api_key:
            st.error("Add your Anthropic API key in the sidebar first.")
        else:
            st.session_state.display_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    client = anthropic.Anthropic(api_key=api_key)
                    answer = run_agent(client, model, prompt)
                st.markdown(answer)
            st.session_state.display_messages.append({"role": "assistant", "content": answer})

# ---- Predictions dashboard tab ----
with tab_predict:
    st.subheader("Manual forecast")
    if not st.session_state.dataframes:
        st.info("Upload a CSV/Excel file in the Ingest tab to enable forecasting.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            ds_name = st.selectbox("Dataset", list(st.session_state.dataframes.keys()))
        numeric_cols = (
            st.session_state.dataframes[ds_name]
            .select_dtypes(include="number")
            .columns.tolist()
        )
        with col2:
            col_name = st.selectbox("Column", numeric_cols) if numeric_cols else None
        with col3:
            periods = st.number_input("Periods ahead", min_value=1, max_value=24, value=3)

        if col_name and st.button("Run forecast"):
            result_text = forecast_column(ds_name, col_name, int(periods))
            st.write(result_text)

    if st.session_state.last_forecast:
        fc = st.session_state.last_forecast
        history = fc["history"]
        forecast = fc["forecast"]
        x_hist = list(range(len(history)))
        x_fc = list(range(len(history), len(history) + len(forecast)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_hist, y=history, mode="lines+markers", name="Actual"))
        fig.add_trace(
            go.Scatter(
                x=[x_hist[-1]] + x_fc,
                y=[history[-1]] + forecast,
                mode="lines+markers",
                name="Forecast",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{fc['column']} — {fc['trend']} trend ({fc['dataset']})",
            xaxis_title="Period",
            yaxis_title=fc["column"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Document risk assessment")
    if st.button("Assess risk in ingested documents"):
        st.write(assess_risk())
