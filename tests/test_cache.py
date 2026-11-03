"""Persistent batch caching without running Chroma or Ollama."""

import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src import rag
from src import cache
from src.cache import append_record, cache_key, index_fingerprint, load_cache
from src.schemas import AuditRecord, AuditVerdict
from src.vector_store import get_embeddings


@pytest.fixture
def services(clause_data, verdict_data):
    # Real settings objects serialize exactly like production; inference is mocked.
    settings = rag.create_audit_client(model="test-model").model_dump(mode="json")
    client = Mock()
    client.model_dump.return_value = settings
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict.model_validate(verdict_data)
    store = Mock()
    store.embeddings = get_embeddings()
    store._collection.metadata = {"hnsw:space": "cosine"}
    store._collection.configuration_json = {"hnsw": {"space": "cosine"}}
    store.get.return_value = {
        "ids": ["swe-050"], "documents": ["Search text"],
        "metadatas": [clause_data.copy()], "embeddings": [[0.1, 0.2]],
    }
    store.similarity_search.return_value = [Document(page_content="Search text", metadata=clause_data)]
    return store, client


def test_persistence_reuse_order_and_current_ids(tmp_path, services, capsys):
    store, client = services
    path = tmp_path / "nested" / "cache.jsonl"
    requirements = [("r3", "First"), ("r1", "Second")]
    original = rag.run_audits(requirements, store, client, k=2, cache_path=path)
    assert len(load_cache(path)) == 2
    assert len(path.read_text().splitlines()) == 2
    store.get.assert_called_once_with(include=["documents", "metadatas", "embeddings"])
    store.similarity_search.reset_mock()
    client.with_structured_output.reset_mock()

    resumed = rag.run_audits(
        [("new-id", "Second"), ("r3", "First"), ("r4", "Third")],
        store, client, k=2, cache_path=path,
    )
    assert [record.requirement_id for record in resumed] == ["new-id", "r3", "r4"]
    assert resumed[0].verdict == original[1].verdict
    store.similarity_search.assert_called_once_with("Third", k=2)
    client.with_structured_output.assert_called_once()
    assert "2 reused; 1 newly audited" in capsys.readouterr().out


@pytest.mark.parametrize("failure_site", ["retrieval", "inference"])
@pytest.mark.parametrize("failure", [ConnectionError("service stopped"), RuntimeError("unexpected")])
def test_resume_after_interruption(tmp_path, services, failure_site, failure):
    store, client = services
    path = tmp_path / "cache.jsonl"
    requirements = [("a", "First"), ("b", "Second")]
    failing_call = (store.similarity_search if failure_site == "retrieval"
                    else client.with_structured_output.return_value.invoke)
    success = failing_call.return_value
    failing_call.side_effect = [success, failure]
    with pytest.raises(type(failure), match=str(failure)):
        rag.run_audits(requirements, store, client, k=2, cache_path=path)
    assert len(load_cache(path)) == 1
    failing_call.side_effect = None
    store.similarity_search.reset_mock()
    client.with_structured_output.reset_mock()
    records = rag.run_audits(requirements, store, client, k=2, cache_path=path)
    assert [record.requirement_id for record in records] == ["a", "b"]
    store.similarity_search.assert_called_once_with("Second", k=2)
    assert len(load_cache(path)) == 2


@pytest.mark.parametrize("failure", ["empty", "invalid", "ungrounded"])
def test_failed_requirements_are_retried(tmp_path, services, failure):
    store, client = services
    path = tmp_path / "cache.jsonl"
    invoke = client.with_structured_output.return_value.invoke
    if failure == "empty":
        documents = store.similarity_search.return_value
        store.similarity_search.side_effect = [[], documents]
    elif failure == "invalid":
        invoke.side_effect = [OutputParserException("bad"), OutputParserException("bad"), invoke.return_value]
    else:
        invalid = invoke.return_value.model_copy(update={"clause_id": "999"})
        invoke.side_effect = [invalid, invalid, invoke.return_value]
    requirements = [("a", "First"), ("b", "Second")]
    records = rag.run_audits(requirements, store, client, k=2, cache_path=path)
    assert records[0].error and records[1].grounded
    assert len(load_cache(path)) == 1
    invoke.side_effect = None
    store.similarity_search.side_effect = None
    store.similarity_search.reset_mock()
    records = rag.run_audits(requirements, store, client, k=2, cache_path=path)
    assert all(record.grounded for record in records)
    store.similarity_search.assert_called_once_with("First", k=2)


@pytest.mark.parametrize("change", [
    "model", "temperature", "embedding_model", "embedding_settings", "prompt",
    "context", "validation", "schema", "k", "text", "document", "metadata", "vector", "id", "metric",
])
def test_configuration_changes_invalidate_cache(tmp_path, services, monkeypatch, change):
    store, client = services
    path = tmp_path / "cache.jsonl"
    rag.run_audits([("a", "First")], store, client, k=2, cache_path=path)
    k, text = 2, "First"
    if change in {"model", "temperature"}:
        settings = client.model_dump(mode="json")
        settings[change] = "other-model" if change == "model" else 0.8
        client.model_dump.return_value = settings
    elif change == "embedding_model":
        store.embeddings = store.embeddings.model_copy(update={"model": "other-embedder"})
    elif change == "embedding_settings":
        store.embeddings = store.embeddings.model_copy(update={"num_ctx": 1000})
    elif change == "prompt":
        monkeypatch.setattr(rag, "SYSTEM_PROMPT", rag.SYSTEM_PROMPT + " New instruction.")
    elif change == "context":
        original = rag.build_user_prompt

        def new_context(text, clauses):
            return original(text, clauses) + "\n"

        monkeypatch.setattr(rag, "build_user_prompt", new_context)
    elif change == "validation":
        original = rag.audit_requirement

        def new_validation(*args, **kwargs):
            return original(*args, **kwargs)

        monkeypatch.setattr(rag, "audit_requirement", new_validation)
    elif change == "schema":
        monkeypatch.setattr(rag.AuditVerdict, "model_json_schema", lambda: {"changed": True})
    elif change == "k":
        k = 3
    elif change == "text":
        text = "Changed requirement"
    elif change == "document":
        store.get.return_value["documents"][0] = "Changed document"
    elif change == "metadata":
        store.get.return_value["metadatas"][0]["requirement"] = "Changed clause"
    elif change == "vector":
        store.get.return_value["embeddings"][0][0] = 0.9
    elif change == "id":
        store.get.return_value["ids"][0] = "other-id"
    elif change == "metric":
        store._collection.configuration_json = {"hnsw": {"space": "l2"}}
    store.similarity_search.reset_mock()
    client.with_structured_output.reset_mock()
    rag.run_audits([("a", text)], store, client, k=k, cache_path=path)
    store.similarity_search.assert_called_once_with(text, k=k)
    client.with_structured_output.assert_called_once()


def test_fingerprint_ignores_row_and_mapping_order(services):
    store, _ = services
    contents = store.get.return_value
    for values in contents.values():
        values.append(values[0])
    contents["ids"][1] = "other-id"
    before = index_fingerprint(store)
    store.get.return_value = {key: list(reversed(value)) for key, value in contents.items()}
    assert index_fingerprint(store) == before
    assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})


def test_disabled_cache_never_reads_settings_or_index(services, capsys):
    store, client = services
    client.model_dump.side_effect = AssertionError("Caching disabled")
    store.get.side_effect = AssertionError("Caching disabled")
    for _ in range(2):
        assert rag.run_audits([("a", "First")], store, client, k=2)[0].grounded
    assert store.similarity_search.call_count == 2
    assert client.with_structured_output.call_count == 2
    assert capsys.readouterr().out == ""


@pytest.fixture
def record(verdict_data):
    return AuditRecord(
        requirement_id="a", requirement_text="First", retrieved_clause_ids=["4.1.2"],
        verdict=AuditVerdict.model_validate(verdict_data), grounded=True,
    )


@pytest.mark.parametrize("tail", [b'{"key":', b'{"key":"cut off', b'{"key":"\xe2\x82'])
def test_torn_tail_is_repaired_before_append(tmp_path, record, tail):
    path = tmp_path / "cache.jsonl"
    append_record(path, "first", record)
    with path.open("ab") as handle:
        handle.write(tail)
    assert list(load_cache(path)) == ["first"]
    append_record(path, "next", record)
    assert list(load_cache(path)) == ["first", "next"]


def test_valid_unterminated_last_line_is_preserved(tmp_path, record):
    path = tmp_path / "cache.jsonl"
    append_record(path, "first", record)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert list(load_cache(path)) == ["first"]
    append_record(path, "next", record)
    assert list(load_cache(path)) == ["first", "next"]


@pytest.mark.parametrize("content,error", [
    ('{bad}\n', json.JSONDecodeError),
    ('{bad}\n{}', json.JSONDecodeError),
    ('{"key":"a","record":{}}', ValidationError),
    ('{"record":{}}', ValueError),
    ('{"key":2,"record":{}}', ValueError),
])
def test_corrupt_complete_entries_are_not_hidden(tmp_path, content, error):
    path = tmp_path / "cache.jsonl"
    path.write_text(content)
    with pytest.raises(error):
        load_cache(path)


@pytest.mark.parametrize("update", [
    {"verdict": None}, {"grounded": False}, {"error": "failed"}, {"retrieved_clause_ids": []},
])
def test_unsuccessful_records_not_persisted_or_reused(tmp_path, record, update):
    path = tmp_path / "cache.jsonl"
    invalid = record.model_copy(update=update)
    append_record(path, "key", invalid)
    assert not path.exists()
    path.write_text(json.dumps({"key": "key", "record": invalid.model_dump(mode="json")}) + "\n")
    assert load_cache(path) == {}


def test_invalid_directories_and_absent_cache(tmp_path, record):
    with pytest.raises(ValueError):
        load_cache(tmp_path)
    with pytest.raises(ValueError):
        append_record(tmp_path, "key", record)
    assert load_cache(tmp_path / "absent.jsonl") == {}


@pytest.mark.parametrize("relative", [
    "results/october/cache.jsonl", "results/october/nested/new.jsonl",
    "data/cache.jsonl", "data/nested/new.jsonl",
])
def test_protected_cache_paths_are_rejected_before_repair_or_append(tmp_path, monkeypatch, record, relative):
    monkeypatch.setattr(cache, "PROJECT_ROOT", tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    original = b'{"key":'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="protected"):
        load_cache(path)
    with pytest.raises(ValueError, match="protected"):
        append_record(path, "key", record)
    assert path.read_bytes() == original
    path.unlink()
    with pytest.raises(ValueError, match="protected"):
        append_record(path, "key", record)
    assert not path.exists()


@pytest.mark.parametrize("alias_type", ["file", "directory"])
def test_symlink_aliases_cannot_bypass_cache_protection(tmp_path, monkeypatch, record, alias_type):
    monkeypatch.setattr(cache, "PROJECT_ROOT", tmp_path)
    history = tmp_path / "results/october"
    history.mkdir(parents=True)
    published = history / "cache.jsonl"
    published.write_bytes(b'{"key":')
    if alias_type == "file":
        path = tmp_path / "alias.jsonl"
        path.symlink_to(published)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(history, target_is_directory=True)
        path = alias / "cache.jsonl"
    with pytest.raises(ValueError, match="protected"):
        load_cache(path)
    with pytest.raises(ValueError, match="protected"):
        append_record(path, "key", record)
    assert published.read_bytes() == b'{"key":'


@pytest.mark.parametrize("name", ["benchmark.csv", "report.md", "metrics.json", "source.py", "input.txt"])
def test_non_cache_files_are_never_repaired_or_appended(tmp_path, record, name):
    path = tmp_path / name
    path.write_text("unrelated input without a newline")
    with pytest.raises(ValueError, match=".jsonl"):
        load_cache(path)
    with pytest.raises(ValueError, match=".jsonl"):
        append_record(path, "key", record)
    assert path.read_text() == "unrelated input without a newline"


def test_cache_destination_cannot_be_a_directory(tmp_path):
    path = tmp_path / "directory.jsonl"
    path.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        load_cache(path)
