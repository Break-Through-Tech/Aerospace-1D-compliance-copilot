"""Benchmark execution with the real pipeline/evaluator and mocked services."""

from unittest.mock import Mock

import pandas as pd
from langchain_core.documents import Document

from scripts import evaluate_rag
from src import rag
from src.evaluation import evaluate
from src.schemas import AuditVerdict


def test_main_loads_resources_once_limits_work_and_evaluates(
    monkeypatch, capsys, clause_data, verdict_data,
):
    benchmark = pd.DataFrame([
        {"requirement_id": req_id, "requirement_text": text, "clause_id": "4.1.2", "label": "Meets"}
        for req_id, text in [("r3", "First requirement"), ("r1", "Second requirement"), ("r2", "Third requirement")]
    ])
    read_csv = Mock(return_value=benchmark)
    monkeypatch.setattr(evaluate_rag.pd, "read_csv", read_csv)
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="Search text", metadata=clause_data)]
    load_store = Mock(return_value=store)
    monkeypatch.setattr(evaluate_rag, "load_vector_store", load_store)
    client = Mock()
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict.model_validate(verdict_data)
    make_client = Mock(return_value=client)
    monkeypatch.setattr(rag, "ChatOllama", make_client)
    evaluate_records = Mock(wraps=evaluate)
    monkeypatch.setattr(evaluate_rag, "evaluate", evaluate_records)

    evaluate_rag.main(["--limit", "2", "--k", "3"])

    load_store.assert_called_once_with()
    make_client.assert_called_once_with(
        model=rag.GENERATION_MODEL,
        base_url=rag.OLLAMA_HOST,
        temperature=0,
        num_ctx=4096,
        num_predict=600,
        reasoning=False,
    )
    assert store.similarity_search.call_count == 2
    assert client.with_structured_output.call_count == 2
    evaluated_benchmark, records = evaluate_records.call_args.args
    assert evaluated_benchmark.requirement_id.tolist() == ["r3", "r1"]
    assert [record.requirement_id for record in records] == ["r3", "r1"]
    assert evaluate_records.call_args.kwargs == {"k": 3}
    output = capsys.readouterr().out
    assert "Evaluated 2 requirements" in output
    assert "Verdict accuracy: 100.00%" in output
