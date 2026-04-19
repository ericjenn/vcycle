"""
rag/pipeline.py
───────────────
Retrieval-Augmented Generation pipeline.

• Loads all PDFs from a configurable directory.
• Chunks text (~1000 tokens), embeds with OpenAI, stores in FAISS.
• Exposes query(text, k=5) → list[str] of relevant passages.

If no PDFs are present the pipeline returns empty results gracefully
so the rest of the system can still run.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _try_import():
    """Lazy imports so the file can be parsed even without all deps installed."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        return PyPDFLoader, RecursiveCharacterTextSplitter, OpenAIEmbeddings, FAISS
    except ImportError as e:
        print(e)
        raise ImportError(
            "RAG dependencies missing. "
            "Run: pip install langchain langchain-community langchain-openai faiss-cpu pypdf"
        ) from e


class RAGPipeline:
    """
    Build and query a FAISS vector store from a directory of PDF files.

    Parameters
    ----------
    docs_dir : str
        Path to the directory containing PDF files.
    chunk_size : int
        Approximate token count per chunk (default 1000).
    chunk_overlap : int
        Overlap between consecutive chunks (default 100).
    top_k : int
        Number of chunks returned per query (default 5).
    """

    def __init__(
        self,
        docs_dir: str = "Docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        top_k: int = 5,
    ):
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._vectorstore: Optional[object] = None
        self._ready = False

    # ── build ────────────────────────────────────────────────────────────────
    def build(self) -> None:
        """Load PDFs, chunk, embed, and index. Call once before querying."""
        PyPDFLoader, RecursiveCharacterTextSplitter, OpenAIEmbeddings, FAISS = (
            _try_import()
        )

        pdf_files = list(self.docs_dir.glob("*.pdf")) if self.docs_dir.exists() else []

        if not pdf_files:
            logger.warning(
                "RAGPipeline: no PDF files found in '%s'. "
                "Queries will return empty results.",
                self.docs_dir,
            )
            self._ready = False
            return

        # Load documents
        all_docs = []
        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf))
                all_docs.extend(loader.load())
                logger.info("RAGPipeline: loaded %s", pdf.name)
            except Exception as exc:
                logger.warning("RAGPipeline: could not load %s — %s", pdf.name, exc)

        if not all_docs:
            logger.warning("RAGPipeline: loaded 0 pages — skipping index build.")
            return

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(all_docs)
        logger.info("RAGPipeline: %d chunks from %d pages.", len(chunks), len(all_docs))

        # Embed + index
        embeddings = OpenAIEmbeddings()
        self._vectorstore = FAISS.from_documents(chunks, embeddings)
        self._ready = True
        logger.info("RAGPipeline: FAISS index built (%d vectors).", len(chunks))

    # ── query ────────────────────────────────────────────────────────────────
    def query(self, text: str, k: Optional[int] = None) -> list[str]:
        """
        Return up to k relevant text passages for *text*.

        Returns an empty list if the index is not ready.
        """
        if not self._ready or self._vectorstore is None:
            return []
        k = k or self.top_k
        results = self._vectorstore.similarity_search(text, k=k)
        return [doc.page_content for doc in results]

    # ── convenience ──────────────────────────────────────────────────────────
    def context_for(self, query: str) -> str:
        """Return retrieved passages joined as a single context string."""
        passages = self.query(query)
        if not passages:
            return ""
        return "\n\n---\n\n".join(passages)
