"""Backward-compatible baseline entry point; implementation lives in src.compliance_rag.

Run from VS Code or a terminal: python scripts/compliance_rag_baseline.py --help
"""
from pathlib import Path
import sys

# Direct script execution adds scripts/, rather than the repository root, to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compliance_rag.schemas import Verdict, Clause, AuditVerdict, AuditRecord
from src.compliance_rag.data import load_clauses, load_benchmark
from src.compliance_rag.embeddings import TfidfEmbedder, SbertEmbedder, OllamaEmbedder
from src.compliance_rag.retrieval import ClauseIndex
from src.compliance_rag.prompts import SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt
from src.compliance_rag.judges import ClaudeJudge, GeminiJudge, OllamaJudge, HeuristicJudge
from src.compliance_rag.cache import _cache_key
from src.compliance_rag.pipeline import run_pipeline
from src.compliance_rag.evaluation import prf, evaluate
from src.compliance_rag.reporting import write_report
from src.compliance_rag.cli import main


if __name__ == "__main__":
    main()
