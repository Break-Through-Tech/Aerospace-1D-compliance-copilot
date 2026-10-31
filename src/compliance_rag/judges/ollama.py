"""OllamaJudge provider."""
import os
from typing import List, Optional
from pydantic import ValidationError
from ..schemas import AuditVerdict, Clause
from ..prompts import SYSTEM_PROMPT, build_user_prompt


class OllamaJudge:
    """Local LLM through Ollama (default qwen3:4b). Output is constrained to AuditVerdict's JSON schema via
    Ollama's `format=` structured outputs, then validated with Pydantic + grounding check, with retry-on-failure."""
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None, max_retries: int = 2,
                 num_ctx: int = 4096):
        from ollama import Client
        self.client = Client(host=host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.max_retries, self.num_ctx = max_retries, num_ctx

    def judge(self, req_text: str, hits: List[tuple[Clause, float]]) -> AuditVerdict:
        allowed = {c.clause_id for c, _ in hits}
        system = SYSTEM_PROMPT.replace("calling the submit_verdict tool", "returning JSON that matches the schema")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": build_user_prompt(req_text, hits)}]
        last_err = None
        for _ in range(self.max_retries + 1):
            resp = self.client.chat(
                model=self.model, messages=messages, format=AuditVerdict.model_json_schema(),
                options={"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 600},
                think=False, stream=False, keep_alive="30m")
            text = resp.message.content
            try:
                v = AuditVerdict.model_validate_json(text)
                if v.clause_id not in allowed:
                    raise ValueError(f"cited clause {v.clause_id!r} not in retrieved set {sorted(allowed)}")
                return v
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:400]
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Invalid output: {last_err}. Allowed clause_id values: "
                                                         f"{sorted(allowed)}. Return corrected JSON only."}]
        raise RuntimeError(f"judge failed after retries: {last_err}")
