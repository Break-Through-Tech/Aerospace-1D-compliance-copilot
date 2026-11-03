"""ClaudeJudge provider."""
import os
import time
from typing import List, Optional
from pydantic import ValidationError
from ..schemas import AuditVerdict, Clause
from ..prompts import SYSTEM_PROMPT, build_user_prompt


class ClaudeJudge:
    """Forced tool-use => the API returns JSON matching AuditVerdict's schema. Validated with Pydantic."""
    name = "claude"

    def __init__(self, model: Optional[str] = None, max_retries: int = 2):
        import anthropic
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.max_retries = max_retries
        self.tool = {
            "name": "submit_verdict",
            "description": "Submit the structured compliance verdict for this requirement.",
            "input_schema": AuditVerdict.model_json_schema(),
        }

    def judge(self, req_text: str, hits: List[tuple[Clause, float]]) -> AuditVerdict:
        allowed = {c.clause_id for c, _ in hits}
        messages = [{"role": "user", "content": build_user_prompt(req_text, hits)}]
        last_err = None
        for attempt in range(self.max_retries + 1):
            resp = self._call(messages)
            block = next((b for b in resp.content if b.type == "tool_use"), None)
            if block is None:
                last_err = "no tool call returned"
            else:
                try:
                    v = AuditVerdict.model_validate(block.input)
                    if v.clause_id not in allowed:
                        raise ValueError(f"cited clause {v.clause_id!r} not in retrieved set {sorted(allowed)}")
                    return v
                except (ValidationError, ValueError) as e:
                    last_err = str(e)
                    messages += [
                        {"role": "assistant", "content": resp.content},
                        {"role": "user", "content": [{
                            "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                            "content": f"Invalid output: {last_err}. Fix it and call submit_verdict again."}]},
                    ]
        raise RuntimeError(f"judge failed after retries: {last_err}")

    def _call(self, messages):
        delay = 2.0
        for _ in range(5):                            # simple backoff for rate limits / overloads
            try:
                return self.client.messages.create(
                    model=self.model, max_tokens=800, temperature=0, system=SYSTEM_PROMPT,
                    tools=[self.tool], tool_choice={"type": "tool", "name": "submit_verdict"},
                    messages=messages)
            except Exception as e:                    # noqa: BLE001
                if "rate" in str(e).lower() or "overloaded" in str(e).lower() or "529" in str(e):
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        raise RuntimeError("API retries exhausted")
