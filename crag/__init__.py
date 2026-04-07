"""
crag/ — Corrective Retrieval-Augmented Generation module.

Phase 2 của Chatbot Y tế: LangGraph-based state machine với
Document Grading, Query Rewriting, và Hallucination Checking.

Ref: Yan et al., "Corrective Retrieval Augmented Generation", ICML 2024
"""

from crag.state import CRAGState
from crag.graph import build_crag_graph
from crag.pipeline import run_crag_pipeline

__all__ = ["CRAGState", "build_crag_graph", "run_crag_pipeline"]
