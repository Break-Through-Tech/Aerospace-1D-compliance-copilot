"""GeminiJudge provider."""
import os
import time
from typing import List, Optional
from pydantic import ValidationError
from ..schemas import AuditVerdict, Clause
from ..prompts import SYSTEM_PROMPT, build_user_prompt


class GeminiJudge:
    """Gemini with JSON-schema-constrained output (response_schema=AuditVerdict), validated with Pydantic.
    Same grounding check + retry-with-feedback loop as ClaudeJudge."""
    name = "gemini"

    def __init__(self, model: Optional[str] = None, max_retries: int = 2):
        from google import genai
        from google.genai import types
        self.types = types
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY)")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.max_retries = max_retries

    def judge(self, req_text: str, hits: List[tuple[Clause, float]]) -> AuditVerdict:
        allowed = {c.clause_id for c, _ in hits}
        prompt = build_user_prompt(req_text, hits)
        last_err = None
        for _ in range(self.max_retries + 1):
            text = self._call(prompt if last_err is None else
                              f"{prompt}\n\nYour previous answer was invalid: {last_err}. Return corrected JSON.")
            try:
                v = AuditVerdict.model_validate_json(text)
                if v.clause_id not in allowed:
                    raise ValueError(f"cited clause {v.clause_id!r} not in retrieved set {sorted(allowed)}")
                return v
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:400]
        raise RuntimeError(f"judge failed after retries: {last_err}")

    def _call(self, prompt: str) -> str:
        cfg = self.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.replace("calling the submit_verdict tool", "returning JSON"),
            temperature=0, response_mime_type="application/json", response_schema=AuditVerdict)
        delay = 5.0
        for attempt in range(5):                      # backoff for per-minute rate limits (429)
            try:
                return self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text
            except Exception as e:                    # noqa: BLE001
                msg = str(e)
                if any(x in msg for x in ("PerDay", "per day", "daily")):
                    raise                              # daily quota: waiting will not help
                if any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "503", "overloaded")):
                    print(f"  rate limited, waiting {delay:.0f}s (attempt {attempt + 1}/5)...", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise
        raise RuntimeError("API retries exhausted (rate limited). Increase --sleep or retry later.")
