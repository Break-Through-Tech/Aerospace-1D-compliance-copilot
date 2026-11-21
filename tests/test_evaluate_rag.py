"""Benchmark execution with the real pipeline/evaluator and mocked services."""

from unittest.mock import Mock, call
import copy
import json

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from scripts import evaluate_rag
from src import agent, rag
from src.evaluation import evaluate
from src.schemas import AuditVerdict
from src.vector_store import get_embeddings


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("use_reporting", [False, True])
@pytest.mark.parametrize("k", [3, 11])
def test_main_loads_resources_once_limits_work_and_evaluates(
    monkeypatch, capsys, clause_data, verdict_data, tmp_path, use_cache, use_reporting, k,
):
    benchmark = pd.DataFrame([
        {"requirement_id": req_id, "requirement_text": text, "clause_id": "4.1.2", "label": "Meets"}
        for req_id, text in [("r3", "First requirement"), ("r1", "Second requirement"), ("r2", "Third requirement")]
    ])
    requirements = list(benchmark[["requirement_id", "requirement_text"]].itertuples(index=False, name=None))
    load_benchmark = Mock(return_value=(benchmark, requirements))
    monkeypatch.setattr(evaluate_rag, "load_benchmark", load_benchmark)
    benchmark_path = tmp_path / "custom-benchmark.csv"
    monkeypatch.setattr(evaluate_rag, "BENCHMARK_PATH", benchmark_path)
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
    run_audits = Mock(wraps=rag.run_audits)
    monkeypatch.setattr(evaluate_rag, "run_audits", run_audits)
    run_agent = Mock()
    monkeypatch.setattr(evaluate_rag, "run_agent", run_agent)

    args = ["--limit", "2", "--k", str(k)]
    if use_cache:
        args += ["--cache", str(tmp_path / "cache.jsonl")]
    if use_reporting:
        args += ["--out", str(tmp_path / "run")]
    evaluate_rag.main(args)

    load_benchmark.assert_called_once_with(benchmark_path)
    run_agent.assert_not_called()
    run_audits.assert_called_once_with(
        requirements[:2], store, client, k=k,
        cache_path=str(tmp_path / "cache.jsonl") if use_cache else None,
    )
    load_store.assert_called_once_with()
    make_client.assert_called_once_with(
        model=rag.GENERATION_MODEL,
        base_url=rag.OLLAMA_HOST,
        temperature=0,
        num_predict=600,
        reasoning=False,
    )
    assert store.similarity_search.call_count == 2
    assert client.with_structured_output.call_count == 2
    evaluated_benchmark, records = evaluate_records.call_args.args
    assert evaluated_benchmark.requirement_id.tolist() == ["r3", "r1"]
    assert [record.requirement_id for record in records] == ["r3", "r1"]
    assert evaluate_records.call_args.kwargs == {"k": k}
    output = capsys.readouterr().out
    assert "Evaluation mode: Canonical RAG" in output
    assert "Evaluated 2 requirements" in output
    assert "Verdict accuracy: 100.00%" in output
    if use_reporting:
        assert "Saved evaluation results to" in output
        saved = json.loads((tmp_path / "run" / "audit_records.json").read_text())
        assert [record["requirement_id"] for record in saved] == ["r3", "r1"]
        assert json.loads((tmp_path / "run" / "metrics.json").read_text())["verdict_accuracy"] == 1.0
        assert "# Canonical RAG evaluation" in (tmp_path / "run" / "report.md").read_text()
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


@pytest.mark.parametrize("limit", [None, 2])
@pytest.mark.parametrize("k", [None, 3])
def test_agent_mode_reuses_services_isolates_runs_and_reports_failures(
    monkeypatch, tmp_path, capsys, clause_data, verdict_data, limit, k,
):
    # Use the real loader and evaluator. Gold data must stop at the loader's pairs.
    benchmark = pd.DataFrame([
        {"requirement_id": req_id, "requirement_text": text, "clause_id": "4.1.2",
         "label": label, "rationale": "REFERENCE RATIONALE MUST NOT REACH AGENT"}
        for req_id, text, label in [
            ("r3", "First requirement", "Gap"),
            ("r1", "Second requirement", "Partial"),
            ("r2", "Third requirement", "Meets"),
        ]
    ])
    benchmark_path = tmp_path / "benchmark.csv"
    benchmark.to_csv(benchmark_path, index=False)
    monkeypatch.setattr(evaluate_rag, "BENCHMARK_PATH", benchmark_path)
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="Search", metadata=clause_data)]
    load_store = Mock(return_value=store)
    monkeypatch.setattr(evaluate_rag, "load_vector_store", load_store)

    def tool_reply(name, arguments, call_id):
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": arguments, "id": call_id, "type": "tool_call"},
        ])

    # Reuse call/retrieval IDs across runs: these must be scoped to each session.
    success = [
        tool_reply("retrieve_clause", {"query": "focused search"}, "search"),
        tool_reply("check_requirement", {"retrieval_id": "retrieval_1"}, "check"),
    ]
    responses = iter([
        [*success, tool_reply("log_gap", {}, "log"), AIMessage(content="Done")],
        [AIMessage(content="Gap")],  # Cannot inherit the first run's verdict/finding.
        [*success, AIMessage(content="Done")],
    ])
    histories = []

    def bind_tools(tools):
        history = []
        histories.append(history)
        replies = iter(next(responses))

        def respond(messages):
            history.append(copy.deepcopy(messages))
            return next(replies)

        return Mock(invoke=Mock(side_effect=respond))

    client = Mock()
    client.bind_tools.side_effect = bind_tools
    client.with_structured_output.return_value.invoke.side_effect = [
        AuditVerdict(**{**verdict_data, "verdict": "Gap", "missing_elements": ["Approval"]}),
        AuditVerdict(**verdict_data),
    ]
    make_client = Mock(return_value=client)
    monkeypatch.setattr(evaluate_rag, "create_audit_client", make_client)
    run_agent = Mock(wraps=agent.run_agent)
    monkeypatch.setattr(evaluate_rag, "run_agent", run_agent)
    run_audits = Mock()
    monkeypatch.setattr(evaluate_rag, "run_audits", run_audits)
    evaluate_records = Mock(wraps=evaluate)
    monkeypatch.setattr(evaluate_rag, "evaluate", evaluate_records)

    out = tmp_path / "agent-run"
    args = ["--agent", "--out", str(out)]
    if limit is not None:
        args += ["--limit", str(limit)]
    if k is not None:
        args += ["--k", str(k)]
    evaluate_rag.main(args)

    selected = benchmark.head(limit) if limit else benchmark
    configured_k = k or 5
    load_store.assert_called_once_with()
    make_client.assert_called_once_with()
    run_audits.assert_not_called()
    assert run_agent.call_args_list == [
        call(row.requirement_id, row.requirement_text, store, client=client, k=configured_k)
        for row in selected.itertuples()
    ]
    assert client.bind_tools.call_count == len(selected)
    for history, row in zip(histories, selected.itertuples()):
        assert len(history[0]) == 2
        assert json.loads(history[0][1].content) == {
            "requirement_id": row.requirement_id, "requirement_text": row.requirement_text,
        }
        assert "REFERENCE RATIONALE" not in history[0][1].content
    assert store.similarity_search.call_args_list == [
        call("focused search", k=configured_k),
    ] * (len(selected) - 1)
    evaluated_benchmark, records = evaluate_records.call_args.args
    assert evaluated_benchmark.requirement_id.tolist() == selected.requirement_id.tolist()
    assert evaluate_records.call_args.kwargs == {"k": configured_k}
    assert records[0].finding_id == "r3:finding_1"
    assert records[1].verdict is None and records[1].finding_id is None and records[1].error
    output = capsys.readouterr().out
    assert "Evaluation mode: Agent" in output
    assert f"Evaluated {len(selected)} requirements (k={configured_k}" in output
    for index, (req_id, status) in enumerate([("r3", "Gap"), ("r1", "FAILED"), ("r2", "Meets")][:len(selected)], 1):
        assert f"[{index}/{len(selected)}] {req_id}: {status}" in output
    saved = json.loads((out / "audit_records.json").read_text())
    assert len(saved) == len(selected)
    assert saved[0]["finding_id"] == "r3:finding_1"
    assert saved[1]["error"] == records[1].error
    predictions = pd.read_csv(out / "predictions.csv", keep_default_na=False)
    assert predictions.pred_verdict.tolist() == ["Gap", "INVALID", "Meets"][:len(selected)]
    assert predictions.iloc[1].audit_error == records[1].error
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n"] == len(selected) and metrics["k"] == configured_k
    assert metrics["confusion_matrix"]["Partial"]["INVALID"] == 1
    assert f"hit_at_{configured_k}" in metrics["retrieval"]
    assert "# Agent evaluation" in (out / "report.md").read_text()
    assert "Failed audits: 1" in (out / "report.md").read_text()


@pytest.mark.parametrize("args, message", [
    (["--agent", "--cache", "cache.jsonl"], "agent caching is not supported"),
    (["--agent", "--k", "11"], "--agent requires --k between 1 and 10"),
    (["--agent", "--k", "0"], "--k must be positive"),
    (["--agent", "--limit", "0"], "--limit must be positive"),
])
def test_invalid_agent_options_fail_before_benchmark_or_service_initialization(monkeypatch, capsys, args, message):
    resources = [Mock(), Mock(), Mock()]
    for name, resource in zip(("load_benchmark", "load_vector_store", "create_audit_client"), resources):
        monkeypatch.setattr(evaluate_rag, name, resource)
    with pytest.raises(SystemExit) as exc:
        evaluate_rag.main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    for resource in resources:
        resource.assert_not_called()


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
    monkeypatch.setattr(evaluate_rag, "load_benchmark", read)
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
