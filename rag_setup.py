"""
Anti-Gravity Financial Agentic System
RAG Setup — ChromaDB ingestion with metadata-filtered retrieval
Uses Google Gemini embeddings via langchain-google-genai.
Supports both .txt and .pdf knowledge base files.
"""

import os
import pathlib
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────
KNOWLEDGE_DIR = pathlib.Path(__file__).parent / "knowledge_base"
CHROMA_DIR = pathlib.Path(__file__).parent / "chroma_db"

# Map each knowledge base file to the agent that should retrieve it
FILE_AGENT_MAP = {
    # ── Original .txt files ────────────────────────────────────
    "financial_order_of_operations.txt": "orchestrator",
    "debt_strategies.txt": "debt_shredder",
    "investment_principles.txt": "investment_scout",
    "insurance_basics.txt": "insurance_expert",

    # ── PDF Books — Orchestrator (general financial literacy) ──
    "The-Financial-Order-of-Operations_v2-1.pdf": "orchestrator",
    "16-05-2021-070111The-Richest-Man-in-Babylon.pdf": "orchestrator",
    "Let's Talk Money PDF.pdf": "orchestrator",
    "I Can Do_RBI.pdf": "orchestrator",
    "You Need a Budget PDF.pdf": "orchestrator",

    # ── PDF Books — Debt Shredder ──────────────────────────────
    "FACTSHEET_Good_Bad_Debt.pdf": "debt_shredder",
    "The Total Money Makeover - Dave Ramsey.pdf": "debt_shredder",
    "How To Get Out Of Debt PDF.pdf": "debt_shredder",
    "How_to_Get_Out_of_Debt,_Stay_Out_of_Debt,_and_Live_Prosperously_06032025_051556_PM.pdf": "debt_shredder",
    "You Need a Budget PDF.pdf": "debt_shredder",

    # ── PDF Books — Insurance Expert ───────────────────────────
    "92167234.pdf": "insurance_expert",
    "ALEnglish.pdf": "insurance_expert",
    "ic-33_key_notes_combined.pdf": "insurance_expert",

    # ── PDF Books — Investment Scout ───────────────────────────
    "THE-INTELLIGENT-INVESTOR.pdf": "investment_scout",
    "Coffee Can Investing PDF.pdf": "investment_scout",
    "common_stocks_and_uncommon_profits_and_other_writings.pdf": "investment_scout",
    "The Most Important Thing by Howard Marks.pdf": "investment_scout",
    "The Simple Path to Wealth PDF.pdf": "investment_scout",
    "L-G-0000569553-0013624054.pdf": "investment_scout",
    "A Random Walk Down Wall Street_ The Time-Tested Strategy for Successful Investing.pdf": "investment_scout",
    
    # ── PDF Books — Final Planner (ET Ecosystem) ────────────────────
    "ET Ecosystem Service Mapping.pdf": "final_planner",
}

# ── Embeddings ─────────────────────────────────────────────────────
def get_embeddings():
    """Return Google Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ── Loader helper ──────────────────────────────────────────────────
def _load_file(filepath: pathlib.Path):
    """Pick the right loader based on file extension."""
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(filepath))
    else:
        return TextLoader(str(filepath), encoding="utf-8")


# ── Ingestion ──────────────────────────────────────────────────────
def ingest_documents(force: bool = False) -> Chroma:
    """
    Load knowledge base files (.txt and .pdf), tag them with agent_role
    metadata, split into chunks, and store in ChromaDB.

    Args:
        force: If True, re-ingest even if the ChromaDB already exists.

    Returns:
        A Chroma vector store instance.
    """
    embeddings = get_embeddings()

    # Return existing store if available
    if CHROMA_DIR.exists() and not force:
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_docs = []
    for filename, agent_role in FILE_AGENT_MAP.items():
        filepath = KNOWLEDGE_DIR / filename
        if not filepath.exists():
            print(f"⚠ Skipping missing file: {filepath}")
            continue

        try:
            loader = _load_file(filepath)
            raw_docs = loader.load()
        except Exception as e:
            print(f"⚠ Error loading {filename}: {e}")
            continue

        chunks = splitter.split_documents(raw_docs)
        for chunk in chunks:
            chunk.metadata["agent_role"] = agent_role
            chunk.metadata["source_file"] = filename

        all_docs.extend(chunks)
        print(f"✓ Ingested {len(chunks):>5} chunks from {filename} → {agent_role}")

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"\n✅ Total {len(all_docs)} chunks stored in ChromaDB at {CHROMA_DIR}")
    return vectorstore


# ── Retrieval ──────────────────────────────────────────────────────
def get_retriever(agent_role: str, k: int = 4):
    """
    Return a retriever filtered by agent_role metadata.

    Args:
        agent_role: One of "orchestrator", "debt_shredder", "investment_scout",
                    "insurance_expert".
        k: Number of documents to retrieve.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"agent_role": agent_role},
        }
    )


def retrieve_context(agent_role: str, query: str, k: int = 4) -> str:
    """
    Convenience function: retrieve and concatenate relevant chunks for a query.

    Returns:
        A single string with all retrieved context separated by dividers.
    """
    retriever = get_retriever(agent_role, k)
    docs = retriever.invoke(query)
    if not docs:
        return "No specific knowledge base context found for this query."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔄 Ingesting knowledge base into ChromaDB...")
    store = ingest_documents(force=False)
    print("\n🧪 Testing retrieval for debt_shredder...")
    context = retrieve_context("debt_shredder", "What is the avalanche method?")
    print(context[:500])
