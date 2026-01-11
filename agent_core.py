import sys
assert sys.version_info[:2] == (3, 11), sys.version

import sqlite3
import re
from typing import List, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import difflib
import nltk

for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

# -----------------------------
# HELPERS
# -----------------------------

def attribute_sentences(answer: str, sources: list):
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(answer)
    attributed = []

    for sent in sentences:
        best = None
        best_score = 0

        for src in sources:
            score = difflib.SequenceMatcher(
                None,
                sent.lower(),
                src["text"].lower()
            ).ratio()

            if score > best_score:
                best_score = score
                best = src

        attributed.append({
            "sentence": sent,
            "source": best,
            "score": best_score
        })

    return attributed

def compute_confidence(sources: list):
    if not sources:
        return 0.0

    scores = [s["score"] for s in sources if s["score"] is not None]
    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 2)

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "mistral:7b-instruct"
EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = "index_store"
DEBUG = False

# -----------------------------
# LlamaIndex global settings
# -----------------------------
Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=360
)

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    timeout=360
)

# -----------------------------
# Load index
# -----------------------------
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=3)

# -----------------------------
# Graph state
# -----------------------------
class GraphState(TypedDict):
    question: str
    history: Annotated[List[BaseMessage], add]
    context: str
    prediction: str
    search_needed: bool
    sources: list

# -----------------------------
# Nodes
# -----------------------------
def retrieve_local(state: GraphState):
    response = query_engine.query(state["question"])

    nodes = response.source_nodes or []

    sources = []
    for n in nodes:
        sources.append({
            "text": n.node.text,
            "file": n.node.metadata.get("file_name"),
            "page": n.node.metadata.get("page_label"),
            "score": round(n.score or 0, 3)
        })

    if response.response.strip():
        return {
            "context": response.response,
            "sources": sources,
            "search_needed": False
        }

    return {
        "context": "",
        "sources": [],
        "search_needed": True
    }



def web_search(state: GraphState):
    if DEBUG:
        print("🌐 Searching the web...")

    search = DuckDuckGoSearchRun()
    try:
        return {"context": search.run(state["question"])}
    except Exception:
        return {"context": "No web results available."}


def generate_answer(state: GraphState):
    prompt = f"""
Answer strictly using the context below.

Context:
{state['context']}

Question:
{state['question']}

Rules:
- Use complete sentences
- If unsure, say so
"""

    response = llm.invoke(prompt)

    return {
        "prediction": response.content,
        "history": [AIMessage(content=response.content)],
        "sources": state["sources"]
    }


def sanitize_text(state: GraphState):
    clean = re.sub(r"[*#_~]", "", state["prediction"])
    return {"prediction": clean}

# -----------------------------
# Build graph
# -----------------------------
conn = sqlite3.connect("bot_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_local)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate_answer)
workflow.add_node("sanitize", sanitize_text)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    lambda x: "web_search" if x["search_needed"] else "generate"
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "sanitize")
workflow.add_edge("sanitize", END)

app = workflow.compile(checkpointer=memory)

# -----------------------------
# Public API
# -----------------------------
def ask(question, history):
    result = app.invoke(
        {"question": question, "history": history},
        config={"configurable": {"thread_id": "ui_user"}}
    )

    spans = attribute_sentences(
        result["prediction"],
        result.get("sources", [])
    )

    confidence = compute_confidence(result.get("sources", []))

    return {
        "answer": result["prediction"],
        "spans": spans,
        "sources": result["sources"],
        "confidence": confidence
    }
