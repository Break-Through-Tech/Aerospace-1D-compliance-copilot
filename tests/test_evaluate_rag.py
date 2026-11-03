"""Benchmark execution with the real pipeline/evaluator and mocked services."""

from unittest.mock import Mock

import pandas as pd
import pytest
from langchain_core.documents import Document

from scripts import evaluate_rag
from src import rag
from src.evaluation import evaluate
from src.schemas import AuditVerdict
from src.vector_store import get_embeddings


@pytest.mark.parametrize("use_cache", [False, True])
def test_main_loads_resources_once_limits_work_and_evaluates(
    monkeypatch, capsys, clause_data, verdict_data, tmp_path, use_cache,
):
    benchmark = pd.DataFrame([
        {"requirement_id": req_id, "requirement_text": text, "clause_id": "4.1.2", "label": "Meets"}
        for req_id, text in [("r3", "First requirement"), ("r1", "Second requirement"), ("r2", "Third requirement")]
    ])
    read_csv = Mock(return_value=benchmark)
    monkeypatch.setattr(evaluate_rag.pd, "read_csv", read_csv)
    store = Mock()
    store.embeddings = get_embeddings()
    store._collection.metadata = None
    store._collection.configuration_json = {}
    store.get.return_value = {
        "ids": ["one"], "documents": ["Search text"],
        "metadatas": [clause_data], "embeddings": [[0.1, 0.2]],
    }
    store.similarity_search.return_value = [Document(page_content="Search text", metadata=clause_data)]
    load_store = Mock(return_value=store)
    monkeypatch.setattr(evaluate_rag, "load_vector_store", load_store)
    client = Mock()
    client.model_dump.return_value = rag.create_audit_client().model_dump(mode="json")
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict.model_validate(verdict_data)
    make_client = Mock(return_value=client)
    monkeypatch.setattr(rag, "ChatOllama", make_client)
    evaluate_records = Mock(wraps=evaluate)
    monkeypatch.setattr(evaluate_rag, "evaluate", evaluate_records)

    args = ["--limit", "2", "--k", "3"]
    if use_cache:
        args += ["--cache", str(tmp_path / "cache.jsonl")]
    evaluate_rag.main(args)

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
    if use_cache:
        assert "0 reused; 2 newly audited" in output
        evaluate_rag.main(args)
        assert store.similarity_search.call_count == 2
        assert client.with_structured_output.call_count == 2
        assert "2 reused; 0 newly audited" in capsys.readouterr().out
