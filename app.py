"""
DocMind — Agentic Document Intelligence Platform
--------------------------------------------------
A multi-purpose AI agent that:
  1. Ingests company documents (PDF / DOCX / TXT) and builds a semantic
     search index (RAG) so it can answer questions grounded in real content.
     Search runs on real sentence embeddings (sentence-transformers + FAISS)
     for strong semantic matching. The embedding model is downloaded and
     cached locally the FIRST time it's needed (requires internet once);
     every run after that loads from the local cache in strict offline
     mode, so a HuggingFace outage or missing internet connection can't
     break the app once it's been run at least once with connectivity.
  2. Ingests structured data (CSV / XLSX) and can forecast numeric trends
     (e.g. revenue, complaints, defect counts) using regression.
  3. Answers qualitative / predictive questions that don't have a single
     factual answer (e.g. "What's the expected outcome of the meeting?",
     "What could be the result of Project X?", "Who seems most committed
     based on the notes?") by gathering broad evidence and reasoning over
     it — explicitly flagged as an inference, not a fact lookup.
  4. Scans ingested documents for risk / compliance language and produces
     a risk score.
  5. Wires all of the above together as *tools* that Claude decides to call
     on its own (Anthropic tool-use / function-calling), so this is a real
     agent loop, not a single hardcoded prompt template.

Run with:  streamlit run app.py

Dependencies (requirements.txt):
    streamlit
    pandas
    numpy
    plotly
    scikit-learn
    sentence-transformers
    faiss-cpu
    pypdf
    python-docx
    openpyxl
    anthropic

Offline behavior:
    The embedding model (all-MiniLM-L6-v2, ~90MB) is cached under
    ~/.cache/huggingface the first time it's loaded. Every subsequent load
    is attempted in strict offline mode (HF_HUB_OFFLINE=1) straight from
    that cache — no network call, no dependency on HuggingFace being up.
    Only the very first run on a machine needs internet.
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import faiss
from sklearn.linear_model import LinearRegression
from pypdf import PdfReader
import docx
import anthropic

# --------------------------------------------------------------------------
# Page / session setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="DocMind — Agentic Doc AI", page_icon="🧠", layout="wide")

MODEL_OPTIONS = ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"]

DEFAULT_STATE = {
    "chunks": [],            # list[str]  text chunks from all docs
    "chunk_sources": [],     # list[str]  which file each chunk came from
    "index": None,           # faiss index over chunk embeddings
    "embedder_error": None,  # set if the embedding model couldn't be loaded
    "dataframes": {},        # name -> pd.DataFrame
    "messages": [],          # chat history (Claude message format)
    "display_messages": [],  # chat history (for rendering)
    "last_forecast": None,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    """Loads the sentence-embedding model.

    Preferred path: a plain local folder at ./models/all-MiniLM-L6-v2,
    produced once by running download_model.py. Loading from a filesystem
    path makes ZERO network calls — no HuggingFace Hub API hit, so no
    rate limit (429) or outage can ever affect it.

    Fallback: if that folder doesn't exist, try the HuggingFace cache in
    strict offline mode, then finally a live online download (which is
    what can trigger a 429 if HF is throttling this IP)."""
    from sentence_transformers import SentenceTransformer
    model_name = "all-MiniLM-L6-v2"

    if os.path.isdir(LOCAL_MODEL_DIR) and os.listdir(LOCAL_MODEL_DIR):
        return SentenceTransformer(LOCAL_MODEL_DIR)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        return SentenceTransformer(model_name)
    except Exception:
        pass  # not cached locally yet — try an online load below

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            "Couldn't load the embedding model. No local model folder was "
            f"found at '{LOCAL_MODEL_DIR}', no HuggingFace cache was found, "
            f"and the online download also failed ({e}). Run "
            "`python download_model.py` once (it retries automatically on "
            "HuggingFace rate limits) — after that this app never touches "
            "the network for embeddings again."
        )


# --------------------------------------------------------------------------
# Document ingestion
# --------------------------------------------------------------------------
def extract_text(uploaded_file) -> str:
    """Extract text from an uploaded file. Never raises — returns '' on failure
    so one bad file can't crash the whole ingest step."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if name.endswith(".docx"):
            d = docx.Document(uploaded_file)
            return "\n".join(p.text for p in d.paragraphs)
        if name.endswith(".txt"):
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
    except Exception as e:
        st.warning(f"Could not read '{uploaded_file.name}': {e}")
        return ""
    return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if overlap >= chunk_size:  # guard against infinite loop on bad params
            break
    return [c for c in chunks if c.strip()]


def rebuild_search_index():
    """(Re)embeds every chunk currently in memory and rebuilds the FAISS
    index. Runs the embedder in offline mode after the first successful
    load, so this never depends on live internet access."""
    if not st.session_state.chunks:
        st.session_state.index = None
        return
    try:
        embedder = load_embedder()
    except Exception as e:
        st.session_state.embedder_error = str(e)
        st.session_state.index = None
        return
    st.session_state.embedder_error = None
    embeddings = embedder.encode(st.session_state.chunks, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    st.session_state.index = index


def add_document_to_index(uploaded_file) -> int:
    text = extract_text(uploaded_file)
    if not text.strip():
        return 0
    new_chunks = chunk_text(text)
    if not new_chunks:
        return 0
    st.session_state.chunks.extend(new_chunks)
    st.session_state.chunk_sources.extend([uploaded_file.name] * len(new_chunks))
    return len(new_chunks)


# --------------------------------------------------------------------------
# Agent tools — plain Python functions the model can invoke
# --------------------------------------------------------------------------
def search_documents(query: str, k: int = 4) -> str:
    """Factual retrieval: top-k chunks most semantically similar to the query."""
    if st.session_state.index is None or not st.session_state.chunks:
        if st.session_state.embedder_error:
            return f"Search is unavailable: {st.session_state.embedder_error}"
        return "No documents have been ingested yet."
    try:
        embedder = load_embedder()
        q_emb = np.asarray(embedder.encode([query]), dtype="float32")
        k = min(k, len(st.session_state.chunks))
        _, ids = st.session_state.index.search(q_emb, k)
    except Exception as e:
        return f"Search failed: {e}"

    results = []
    for i in ids[0]:
        if 0 <= i < len(st.session_state.chunks):
            src = st.session_state.chunk_sources[i]
            results.append(f"[source: {src}]\n{st.session_state.chunks[i]}")
    return "\n\n---\n\n".join(results) if results else "No relevant passages found."


def predict_qualitative(question: str, k: int = 8) -> str:
    """Broad retrieval for qualitative / predictive questions that don't have
    a single factual answer (expected outcomes, likely results, who seems
    most committed, sentiment, etc.). Pulls more context than a normal
    factual lookup so the model has enough material to reason across
    multiple passages, and explicitly labels the result as evidence to be
    synthesized into an inference — not a fact to repeat verbatim."""
    if st.session_state.index is None or not st.session_state.chunks:
        return ("No documents have been ingested yet, so there is no evidence "
                 "available to base a prediction on.")
    evidence = search_documents(question, k=k)
    if evidence in ("No relevant passages found.", "No documents have been ingested yet."):
        return evidence + " Not enough evidence to support a reasoned prediction."
    return (
        "EVIDENCE GATHERED (broad retrieval, for reasoning — this is NOT a "
        "single factual answer). Synthesize across these passages, then give "
        "an answer that: (1) is explicitly framed as an inference/prediction, "
        "not a documented fact, (2) states your confidence (low/medium/high) "
        "and why, (3) says plainly if the evidence is too thin to support a "
        "confident judgment.\n\n" + evidence
    )


def forecast_column(dataset_name: str, column_name: str, periods_ahead: int = 3) -> str:
    try:
        df = st.session_state.dataframes.get(dataset_name)
        if df is None:
            available = list(st.session_state.dataframes.keys())
            return f"Dataset '{dataset_name}' not found. Available datasets: {available}"
        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {list(df.columns)}"

        col = df[column_name]
        if isinstance(col, pd.DataFrame):
            # duplicate column names in the source file — take the first
            col = col.iloc[:, 0]

        series = pd.to_numeric(col, errors="coerce").dropna()
        if len(series) < 3:
            return "Not enough numeric data points in that column to forecast (need at least 3)."

        X = np.arange(len(series)).reshape(-1, 1)
        y = series.to_numpy(dtype=float)
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
    except Exception as e:
        return f"Forecast failed due to an unexpected error: {e}"


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
            "relevant to a FACTUAL question — something with a definite, findable "
            "answer in the text. Always use this before answering any factual "
            "question about document content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "predict_qualitative",
        "description": (
            "Gather broad supporting evidence from ingested documents to reason about "
            "ANY qualitative or predictive question that does NOT have a single factual "
            "answer sitting in the text — i.e. anything requiring judgment, forecasting "
            "a non-numeric outcome, assessing likelihood, comparing people/options, or "
            "synthesizing a conclusion across multiple passages. This is a broad "
            "category, not a fixed list — examples include (not limited to): 'What is "
            "the expected outcome of the meeting?', 'What could be the result of "
            "Project X?', 'Who seems most committed to the work based on the notes?', "
            "'How is this negotiation likely to go?', 'Which vendor looks like the "
            "better bet?', 'What risks could derail this timeline?', 'Is the team "
            "aligned or is there hidden disagreement?'. Use this instead of "
            "search_documents whenever the user is asking you to judge, predict, "
            "compare, or infer something rather than look up a stated fact. After "
            "calling this, give an answer explicitly framed as an inference (not a "
            "documented fact), state your confidence level, and say plainly if the "
            "evidence is too thin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The qualitative/predictive question"},
                "k": {"type": "integer", "description": "How many passages to retrieve", "default": 8},
            },
            "required": ["question"],
        },
    },
    {
        "name": "forecast_column",
        "description": (
            "Forecast future NUMERIC values of a column from an uploaded structured "
            "dataset (CSV/Excel) using linear trend regression. Use this whenever the "
            "user asks about future numeric values, projections, or trends in "
            "quantitative data (revenue, counts, scores, etc.)."
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
    "predict_qualitative": lambda inp: predict_qualitative(inp["question"], inp.get("k", 8)),
    "forecast_column": lambda inp: forecast_column(
        inp["dataset_name"], inp["column_name"], inp.get("periods_ahead", 3)
    ),
    "assess_risk": lambda inp: assess_risk(),
}

SYSTEM_PROMPT = (
    "You are DocMind, an enterprise AI agent. You help employees find answers inside "
    "ingested company documents, make data-driven numeric predictions from structured "
    "datasets, and reason about qualitative/predictive questions (expected outcomes, "
    "likely results, who seems most committed, sentiment, etc.).\n\n"
    "Tool selection rules:\n"
    "- Factual question with a definite answer in the documents -> search_documents.\n"
    "- Qualitative/predictive question with no single factual answer (expected "
    "outcome, likely result, who seems more committed, how something will probably "
    "go) -> predict_qualitative, then clearly label your answer as an inference, "
    "give a confidence level, and say if evidence is too thin to judge.\n"
    "- Numeric trend/projection from a dataset -> forecast_column.\n"
    "- Risk/compliance question -> assess_risk.\n\n"
    "Never invent document content. State clearly which tool's findings your answer "
    "relies on. Be concise and precise, and say plainly when there isn't enough "
    "information to answer."
)


def run_agent(client: anthropic.Anthropic, model: str, user_message: str) -> str:
    st.session_state.messages.append({"role": "user", "content": user_message})
    messages = st.session_state.messages

    for _ in range(6):  # cap tool-use loop iterations for safety
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            return f"API call failed: {e}"

        if response.stop_reason == "tool_use":
            # Use the SDK's own serialization so every block type (text,
            # tool_use, thinking, etc.) round-trips correctly — a hand-rolled
            # serializer that only knows about "text"/"tool_use" will silently
            # drop other block types and break the next API call.
            assistant_content = [b.model_dump() for b in response.content]
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = TOOL_FUNCS[block.name](block.input)
                    except Exception as e:
                        result = f"Tool '{block.name}' raised an error: {e}"
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
    st.title("DocMind")
    st.caption("Agentic document QA + predictive analytics (fully offline search)")
    api_key = st.text_input("Anthropic API key", type="password")
    model = st.selectbox("Model", MODEL_OPTIONS, index=0)
    st.divider()
    st.markdown("**Ingested documents:** " + str(len(set(st.session_state.chunk_sources))))
    st.markdown("**Chunks indexed:** " + str(len(st.session_state.chunks)))
    st.markdown("**Datasets loaded:** " + str(len(st.session_state.dataframes)))
    if st.session_state.embedder_error:
        st.error(st.session_state.embedder_error)
    if st.button("Reset session"):
        for key, default in DEFAULT_STATE.items():
            st.session_state[key] = default
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
        with st.spinner("Chunking and indexing..."):
            total = 0
            for f in doc_files:
                total += add_document_to_index(f)
            rebuild_search_index()
        if total:
            st.success(f"Ingested {len(doc_files)} file(s) into {total} searchable chunks.")
        else:
            st.warning("No extractable text found in the uploaded file(s).")

    st.divider()
    st.subheader("Upload structured data for predictions")
    data_files = st.file_uploader(
        "CSV or Excel", type=["csv", "xlsx"], accept_multiple_files=True, key="data_upload"
    )
    if data_files and st.button("Load datasets"):
        loaded = 0
        for f in data_files:
            try:
                if f.name.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(f)
                    except UnicodeDecodeError:
                        f.seek(0)
                        df = pd.read_csv(f, encoding="latin-1")
                else:
                    df = pd.read_excel(f)
                st.session_state.dataframes[f.name] = df
                loaded += 1
            except Exception as e:
                st.warning(f"Could not load '{f.name}': {e}")
        if loaded:
            st.success(f"Loaded {loaded} dataset(s).")

    if st.session_state.dataframes:
        st.subheader("Loaded datasets preview")
        for name, df in st.session_state.dataframes.items():
            with st.expander(name):
                st.dataframe(df.head(20))

# ---- Chat tab ----
with tab_chat:
    st.subheader("Ask DocMind")
    st.caption(
        "Ask factual questions about your documents, request a numeric forecast, "
        "ask a qualitative/predictive question (e.g. 'what's the likely outcome of "
        "the meeting?'), or ask for a risk assessment. The agent decides which "
        "tool(s) to use."
    )

    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "e.g. 'Who seems most committed based on the meeting notes?' or 'Forecast next quarter revenue'"
    )
    if prompt:
        if not api_key:
            st.error("Add your Anthropic API key in the sidebar first.")
        else:
            st.session_state.display_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        answer = run_agent(client, model, prompt)
                    except Exception as e:
                        answer = f"Something went wrong: {e}"
                st.markdown(answer)
            st.session_state.display_messages.append({"role": "assistant", "content": answer})

# ---- Predictions dashboard tab ----
with tab_predict:
    st.subheader("Manual numeric forecast")
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

        if not numeric_cols:
            st.info("No numeric columns detected in this dataset.")
        elif col_name and st.button("Run forecast"):
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
    st.subheader("Qualitative / predictive question")
    st.caption(
        "For questions without a single numeric answer, e.g. 'What's the expected "
        "outcome of Project X?' — use the Chat Agent tab, which calls "
        "predict_qualitative and reasons over the retrieved evidence."
    )

    st.divider()
    st.subheader("Document risk assessment")
    if st.button("Assess risk in ingested documents"):
        st.write(assess_risk())
