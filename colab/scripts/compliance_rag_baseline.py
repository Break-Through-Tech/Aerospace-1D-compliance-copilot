"""Backward-compatible baseline entry point; implementation lives in colab.compliance_rag.

Run from VS Code or a terminal: python colab/scripts/compliance_rag_baseline.py --help
"""
from pathlib import Path
import sys

# Direct script execution adds colab/scripts/, rather than the repository root, to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from colab.compliance_rag.schemas import Verdict, Clause, AuditVerdict, AuditRecord
from colab.compliance_rag.data import load_clauses, load_benchmark
from colab.compliance_rag.embeddings import TfidfEmbedder, SbertEmbedder, OllamaEmbedder
from colab.compliance_rag.retrieval import ClauseIndex
from colab.compliance_rag.prompts import SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt
from colab.compliance_rag.judges import ClaudeJudge, GeminiJudge, OllamaJudge, HeuristicJudge
from colab.compliance_rag.cache import _cache_key
from colab.compliance_rag.pipeline import run_pipeline
from colab.compliance_rag.evaluation import prf, evaluate
from colab.compliance_rag.reporting import write_report
from colab.compliance_rag.cli import main


if __name__ == "__main__":
    main()
