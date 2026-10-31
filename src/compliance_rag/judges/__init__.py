"""Judges with a common judge(requirement_text, hits) interface."""
from .claude import ClaudeJudge
from .gemini import GeminiJudge
from .ollama import OllamaJudge
from .heuristic import HeuristicJudge

__all__ = ["ClaudeJudge", "GeminiJudge", "OllamaJudge", "HeuristicJudge"]
