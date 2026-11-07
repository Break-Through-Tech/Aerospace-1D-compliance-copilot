"""Benchmark execution with the real pipeline/evaluator and mocked services."""

from unittest.mock import Mock
import json

import pandas as pd
import pytest
from langchain_core.documents import Document

from scripts import evaluate_rag
from src import rag
from src.evaluation import evaluate
from src.schemas import AuditVerdict
from src.vector_store import get_embeddings


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("use_reporting", [False, True])
def test_main_loads_resources_once_limits_work_and_evaluates(
    monkeypatch, capsys, clause_data, verdict_data, tmp_path, use_cache, use_reporting,
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
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    client.model_dump.return_value = rag.create_audit_client().model_dump(mode="json")
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict.model_validate(verdict_data)
    make_client = Mock(return_value=client)
    monkeypatch.setattr(rag, "ChatOllama", make_client)
    evaluate_records = Mock(wraps=evaluate)
    monkeypatch.setattr(evaluate_rag, "evaluate", evaluate_records)

    args = ["--limit", "2", "--k", "3"]
    if use_cache:
        args += ["--cache", str(tmp_path / "cache.jsonl")]
    if use_reporting:
        args += ["--out", str(tmp_path / "run")]
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
    if use_reporting:
        assert "Saved evaluation results to" in output
        saved = json.loads((tmp_path / "run" / "audit_records.json").read_text())
        assert [record["requirement_id"] for record in saved] == ["r3", "r1"]
        assert json.loads((tmp_path / "run" / "metrics.json").read_text())["verdict_accuracy"] == 1.0
    else:
        assert "Saved evaluation results" not in output
        assert not (tmp_path / "run").exists()
    if use_cache:
        assert "0 reused; 2 newly audited" in output
        if use_reporting:
            # Keep the first report; give the resumed run its own destination.
            args[-1] = str(tmp_path / "resumed")
        evaluate_rag.main(args)
        assert store.similarity_search.call_count == 2
        assert client.with_structured_output.call_count == 2
        assert "2 reused; 0 newly audited" in capsys.readouterr().out
        if use_reporting:
            assert json.loads((tmp_path / "resumed" / "metrics.json").read_text()) == json.loads(
                (tmp_path / "run" / "metrics.json").read_text()
            )


@pytest.mark.parametrize("destination", ["history", "benchmark", "existing", "cache-conflict"])
def test_invalid_output_fails_before_loading_services(monkeypatch, tmp_path, destination):
    from src import reporting

    monkeypatch.setattr(reporting, "PROJECT_ROOT", tmp_path)
    benchmark = tmp_path / "data" / "benchmark.csv"
    benchmark.parent.mkdir()
    benchmark.write_text("benchmark input")
    history = tmp_path / "results" / "october"
    history.mkdir(parents=True)
    published = history / "metrics.json"
    published.write_text("historical metrics")
    out = {"history": history, "benchmark": benchmark, "existing": tmp_path / "existing",
           "cache-conflict": tmp_path / "new-run"}[destination]
    args = ["--out", str(out)]
    if destination == "existing":
        out.mkdir()
        (out / "report.md").write_text("existing input")
    if destination == "cache-conflict":
        args += ["--cache", str(out / "report.md" / "cache.jsonl")]
    load = Mock()
    monkeypatch.setattr(evaluate_rag, "load_vector_store", load)
    with pytest.raises(SystemExit) as exc:
        evaluate_rag.main(args)
    assert exc.value.code == 2
    load.assert_not_called()
    assert benchmark.read_text() == "benchmark input"
    assert published.read_text() == "historical metrics"
    if destination == "existing":
        assert (out / "report.md").read_text() == "existing input"
    if destination == "cache-conflict":
        assert not out.exists()


@pytest.mark.parametrize("with_output", [False, True])
@pytest.mark.parametrize("relative", ["results/october/cache.jsonl", "data/cache.jsonl", "source.py"])
def test_invalid_cache_fails_before_reading_benchmark_or_loading_services(
    monkeypatch, tmp_path, with_output, relative,
):
    from src import cache

    monkeypatch.setattr(cache, "PROJECT_ROOT", tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"key":')
    read = Mock()
    load = Mock()
    monkeypatch.setattr(evaluate_rag.pd, "read_csv", read)
    monkeypatch.setattr(evaluate_rag, "load_vector_store", load)
    args = ["--cache", str(path)]
    if with_output:
        args += ["--out", str(tmp_path / "run")]
    with pytest.raises(SystemExit) as exc:
        evaluate_rag.main(args)
    assert exc.value.code == 2
    read.assert_not_called()
    load.assert_not_called()
    assert path.read_bytes() == b'{"key":'
    assert not (tmp_path / "run").exists()
