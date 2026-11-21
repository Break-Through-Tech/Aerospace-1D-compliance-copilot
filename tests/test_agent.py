"""Real LangChain tools/messages and canonical RAG with offline stores/clients."""

import copy
import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from src import agent, rag
from src.schemas import AuditVerdict


@pytest.fixture
def setup(clause_data, verdict_data):
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="Search text", metadata=clause_data)]
    client = Mock()
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict(**verdict_data)
    return store, client


def invoke(session, name, **arguments):
    return session.tools[name].invoke(arguments)


def retrieve(session, query="focused search"):
    return invoke(session, "retrieve_clause", query=query)["retrieval_id"]


def assess(session):
    return invoke(session, "check_requirement", retrieval_id=retrieve(session))


def call(name, arguments=None, call_id="call_1"):
    return {"name": name, "args": arguments or {}, "id": call_id, "type": "tool_call"}


def reply(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def script(client, responses):
    """Capture independent message histories; invoke receives a mutable list."""
    histories = []
    responses = iter(responses)

    def respond(messages):
        histories.append(copy.deepcopy(messages))
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    client.bind_tools.return_value.invoke.side_effect = respond
    return histories


def workflow(log=False):
    responses = [
        reply(call("retrieve_clause", {"query": "focused search"})),
        reply(call("check_requirement", {"retrieval_id": "retrieval_1"}, "call_2")),
    ]
    if log:
        responses.append(reply(call("log_gap", call_id="call_3")))
    return [*responses, AIMessage(content="Finished")]


def test_tools_use_canonical_pipeline_and_original_requirement(setup):
    store, client = setup
    session = agent.AuditTools("r1", "Original requirement", store, client)
    result = assess(session)
    store.similarity_search.assert_called_once_with("focused search", k=5)
    client.with_structured_output.assert_called_once_with(AuditVerdict, method="json_schema")
    prompt = client.with_structured_output.return_value.invoke.call_args.args[0][1]["content"]
    assert "Original requirement" in prompt
    assert "focused search" not in prompt
    assert result["grounded"] and result["verdict"]["verdict"] == "Meets"
    result["verdict"]["verdict"] = "Gap"
    assert session.result().verdict.verdict == "Meets"


def test_langchain_generates_named_tool_schemas(setup):
    session = agent.AuditTools("r1", "Original", *setup)
    schemas = {name: convert_to_openai_tool(tool)["function"] for name, tool in session.tools.items()}
    assert {name: schema["name"] for name, schema in schemas.items()} == {
        name: name for name in ("retrieve_clause", "check_requirement", "log_gap")
    }
    assert all(schema["parameters"]["additionalProperties"] is False for schema in schemas.values())
    assert set(schemas["retrieve_clause"]["parameters"]["properties"]) == {"query"}
    assert schemas["check_requirement"]["parameters"]["required"] == ["retrieval_id"]
    assert schemas["log_gap"]["parameters"]["properties"] == {}


@pytest.mark.parametrize("name,args", [
    ("retrieve_clause", {"query": " "}),
    ("retrieve_clause", {"query": "x", "k": 0}),
    ("retrieve_clause", {"query": "x", "k": 11}),
    ("retrieve_clause", {"query": "x", "k": "5"}),
    ("retrieve_clause", {"query": "x", "k": True}),
    ("retrieve_clause", {"query": "x", "k": 5}),
    ("retrieve_clause", {"query": "x", "extra": True}),
    ("retrieve_clause", {"query": "x" * 20001}),
    ("check_requirement", {"retrieval_id": " "}),
    ("check_requirement", {"retrieval_id": "retrieval_1", "requirement_id": "other"}),
    ("check_requirement", {"retrieval_id": "retrieval_1", "verdict": "Meets"}),
    ("log_gap", {"assessment_id": "invented"}),
    ("log_gap", {"verdict": "Gap"}),
])
def test_strict_arguments_reject_invalid_requests_before_work(setup, name, args):
    store, client = setup
    session = agent.AuditTools("r1", "Original", store, client)
    with pytest.raises(ValidationError):
        session.tools[name].invoke(args)
    store.similarity_search.assert_not_called()
    client.with_structured_output.assert_not_called()


def test_unknown_retrieval_and_missing_assessment_rejected(setup):
    session = agent.AuditTools("r1", "Original", *setup)
    with pytest.raises(ValueError, match="Unknown retrieval"):
        invoke(session, "check_requirement", retrieval_id="invented")
    with pytest.raises(ValueError, match="No validated assessment"):
        invoke(session, "log_gap")
    assert session.result().error


def test_chosen_retrieval_cannot_be_replaced_by_later_search(setup, clause_data, verdict_data):
    store, client = setup
    second = {**clause_data, "clause": "4.1.3"}
    store.similarity_search.side_effect = [
        [Document(page_content="first", metadata=clause_data)],
        [Document(page_content="second", metadata=second)],
    ]
    session = agent.AuditTools("r1", "Original", store, client)
    first_id = retrieve(session)
    retrieve(session)
    with pytest.raises(ValueError, match="Retrieval limit"):
        retrieve(session)
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict(**verdict_data)
    result = invoke(session, "check_requirement", retrieval_id=first_id)
    assert result["retrieved_clause_ids"] == ["4.1.2"]
    with pytest.raises(ValueError, match="already attempted"):
        invoke(session, "check_requirement", retrieval_id=first_id)
    with pytest.raises(ValueError, match="precede assessment"):
        retrieve(session)


@pytest.mark.parametrize("label", ["Partial", "Gap"])
def test_logging_keeps_validated_assessment_isolated_and_idempotent(setup, verdict_data, label):
    store, client = setup
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict(
        **{**verdict_data, "verdict": label, "missing_elements": ["Approval"]},
    )
    session = agent.AuditTools("r1", "Original", store, client)
    assessment = assess(session)
    assert session.result().error and session.result().verdict is None
    first = invoke(session, "log_gap")
    second = invoke(session, "log_gap")
    assert not first["already_logged"] and second["already_logged"]
    assert first["finding_id"] == second["finding_id"]
    result = session.result()
    assert result.verdict.model_dump() == assessment["verdict"]
    result.verdict.missing_elements.clear()
    assert session.result().verdict.missing_elements == ["Approval"]
    other = agent.AuditTools("r2", "Different", store, client)
    assert other.result().verdict is None and other.result().finding_id is None
    with pytest.raises(ValueError, match="No validated assessment"):
        invoke(other, "log_gap")
    with pytest.raises(ValueError, match="Unknown retrieval"):
        invoke(other, "check_requirement", retrieval_id="retrieval_1")


def test_meets_cannot_be_logged(setup):
    session = agent.AuditTools("r1", "Original", *setup)
    assess(session)
    with pytest.raises(ValueError, match="Only Partial and Gap"):
        invoke(session, "log_gap")
    assert session.result().finding_id is None


@pytest.mark.parametrize("arguments", [{"assessment_id": "invented"}, {"verdict": "Gap"}])
@pytest.mark.parametrize("as_tool_call", [False, True])
def test_invalid_log_arguments_cannot_create_a_finding(setup, verdict_data, arguments, as_tool_call):
    store, client = setup
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict(
        **{**verdict_data, "verdict": "Gap", "missing_elements": ["Approval"]},
    )
    session = agent.AuditTools("r1", "Original", store, client)
    assess(session)
    tool_input = call("log_gap", arguments) if as_tool_call else arguments
    with pytest.raises(ValidationError):
        session.tools["log_gap"].invoke(tool_input)
    assert session.result().finding_id is None
    assert session.result().error
    assert not invoke(session, "log_gap")["already_logged"]
    assert session.result().error is None


@pytest.mark.parametrize("field,value", [("clause_id", "9.9"), ("evidence", "Invented quote")])
def test_canonical_grounding_failures_never_become_assessments(setup, verdict_data, field, value):
    store, client = setup
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict(
        **{**verdict_data, field: value},
    )
    session = agent.AuditTools("r1", "Original", store, client)
    retrieval_id = retrieve(session)
    with pytest.raises(rag.CitationValidationError):
        invoke(session, "check_requirement", retrieval_id=retrieval_id)
    assert client.with_structured_output.return_value.invoke.call_count == 2
    with pytest.raises(ValueError, match="already attempted"):
        invoke(session, "check_requirement", retrieval_id=retrieval_id)
    assert session.result().verdict is None


def test_canonical_retry_can_correct_an_invalid_verdict(setup, verdict_data):
    store, client = setup
    client.with_structured_output.return_value.invoke.side_effect = [
        AuditVerdict(**{**verdict_data, "evidence": "Invented"}), AuditVerdict(**verdict_data),
    ]
    session = agent.AuditTools("r1", "Original", store, client)
    assert assess(session)["grounded"]
    assert client.with_structured_output.return_value.invoke.call_count == 2


@pytest.mark.parametrize("failure", [[], RuntimeError("Search failed")])
def test_retrieval_failures_remain_failures(setup, failure):
    store, client = setup
    if isinstance(failure, Exception):
        store.similarity_search.side_effect = failure
    else:
        store.similarity_search.return_value = failure
    session = agent.AuditTools("r1", "Original", store, client)
    with pytest.raises((ValueError, RuntimeError)):
        retrieve(session)
    client.with_structured_output.assert_not_called()
    assert session.result().error and session.result().verdict is None


def test_runtime_binds_tools_and_preserves_tool_messages(setup, verdict_data):
    store, client = setup
    histories = script(client, workflow())
    result = agent.run_agent("r1", "Original", store, client=client)
    assert result.error is None and result.grounded
    assert result.verdict.model_dump() == verdict_data
    assert [tool.name for tool in client.bind_tools.call_args.args[0]] == [
        "retrieve_clause", "check_requirement", "log_gap",
    ]
    assert isinstance(histories[0][0], SystemMessage)
    assert json.loads(histories[0][1].content)["requirement_text"] == "Original"
    assert [len(history) for history in histories] == [2, 4, 6]
    for history, expected_id in [(histories[1], "call_1"), (histories[2], "call_2")]:
        assert isinstance(history[-1], ToolMessage)
        assert history[-1].tool_call_id == expected_id
        assert json.loads(history[-1].content)["ok"]


@pytest.mark.parametrize("log", [False, True])
def test_runtime_requires_gap_logging(setup, verdict_data, log):
    store, client = setup
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict(
        **{**verdict_data, "verdict": "Gap", "missing_elements": ["Approval"]},
    )
    script(client, workflow(log))
    result = agent.run_agent("r1", "Original", store, client=client)
    if log:
        assert result.error is None and result.finding_id == "r1:finding_1"
    else:
        assert "logging" in result.error and result.verdict is None and not result.grounded


def test_model_final_text_cannot_fabricate_or_overwrite_verdict(setup, verdict_data):
    store, client = setup
    fabricated = AIMessage(content=json.dumps({**verdict_data, "verdict": "Gap"}))
    script(client, [fabricated])
    result = agent.run_agent("r1", "Original", store, client=client)
    assert result.verdict is None and result.error
    script(client, [*workflow()[:-1], fabricated])
    result = agent.run_agent("r1", "Original", store, client=client)
    assert result.verdict.model_dump() == verdict_data


@pytest.mark.parametrize("bad_call", [
    call("unknown"),
    call("retrieve_clause", {"query": "x", "k": "5"}),
    call("check_requirement", {"retrieval_id": "invented"}),
])
def test_tool_errors_return_to_model_and_allow_correction(setup, bad_call):
    store, client = setup
    bad_call["id"] = "bad_call"
    histories = script(client, [reply(bad_call), *workflow()])
    result = agent.run_agent("r1", "Original", store, client=client)
    assert result.error is None
    error = histories[1][-1]
    assert error.tool_call_id == "bad_call" and error.status == "error"
    assert json.loads(error.content)["ok"] is False
    assert "verdict" not in json.loads(error.content)


def test_multiple_calls_in_a_turn_are_processed_in_order(setup):
    store, client = setup
    histories = script(client, [
        reply(call("retrieve_clause", {"query": "x"}),
              call("check_requirement", {"retrieval_id": "retrieval_1"}, "call_2")),
        AIMessage(content="Finished"),
    ])
    result = agent.run_agent("r1", "Original", store, client=client)
    assert result.error is None
    assert [message.tool_call_id for message in histories[1][-2:]] == ["call_1", "call_2"]


def test_repeated_failures_terminate_at_turn_limit(setup):
    store, client = setup
    histories = script(client, [reply(call("unknown", call_id=f"call_{i}")) for i in range(3)])
    result = agent.run_agent("r1", "Original", store, client=client, max_steps=3)
    assert "model-turn limit" in result.error and result.verdict is None
    assert len(histories) == 3
    store.similarity_search.assert_not_called()


def test_tool_call_budget_rejects_oversized_batch_before_execution(setup):
    store, client = setup
    script(client, [reply(*(call("retrieve_clause", {"query": "x"}, f"call_{i}") for i in range(3)))])
    result = agent.run_agent("r1", "Original", store, client=client, max_steps=2)
    assert "tool-call limit" in result.error
    store.similarity_search.assert_not_called()


@pytest.mark.parametrize("response", [
    "not an AIMessage",
    AIMessage(content="", invalid_tool_calls=[{
        "name": "retrieve_clause", "args": "{broken", "id": "bad", "error": "Invalid JSON",
    }]),
    reply(call("retrieve_clause", {"query": "x"}, "")),
    reply(call("retrieve_clause", {"query": "x"}), call("log_gap")),
])
def test_invalid_model_responses_fail_explicitly(setup, response):
    store, client = setup
    script(client, [response])
    result = agent.run_agent("r1", "Original", store, client=client)
    assert result.error and result.verdict is None
    store.similarity_search.assert_not_called()


def test_reused_tool_call_id_fails_without_duplicate_work(setup):
    store, client = setup
    script(client, [reply(call("retrieve_clause", {"query": "x"}))] * 2)
    result = agent.run_agent("r1", "Original", store, client=client)
    assert "unique" in result.error
    store.similarity_search.assert_called_once()


@pytest.mark.parametrize("stage", ["bind", "model", "audit"])
def test_service_failures_never_become_compliance_verdicts(setup, stage):
    store, client = setup
    if stage == "bind":
        client.bind_tools.side_effect = RuntimeError("Tools unsupported")
    elif stage == "model":
        script(client, [ConnectionError("Ollama unavailable")])
    else:
        client.with_structured_output.return_value.invoke.side_effect = ConnectionError("Ollama unavailable")
        histories = script(client, workflow())
    result = agent.run_agent("r1", "Original", store, client=client)
    assert result.error and result.verdict is None and not result.grounded
    if stage == "audit":
        assert "Ollama unavailable" in json.loads(histories[-1][-1].content)["error"]
        client.with_structured_output.return_value.invoke.assert_called_once()


def test_runtime_state_is_fresh_for_every_requirement(setup):
    store, client = setup
    script(client, workflow())
    assert agent.run_agent("r1", "First", store, client=client).verdict is not None
    histories = script(client, [AIMessage(content="Meets")])
    result = agent.run_agent("r2", "Second", store, client=client)
    assert result.requirement_id == "r2" and result.requirement_text == "Second"
    assert result.verdict is None and result.finding_id is None and result.retrieved_clause_ids == []
    assert len(histories[0]) == 2


@pytest.mark.parametrize("k", [1, 3, 10])
def test_configured_k_applies_to_reformulated_searches_and_chosen_retrieval(setup, clause_data, k):
    store, client = setup
    store.similarity_search.side_effect = [
        [Document(page_content="First", metadata={**clause_data, "clause": "other"})],
        [Document(page_content="Second", metadata=clause_data)],
    ]
    histories = script(client, [
        reply(call("retrieve_clause", {"query": "original query"}, "search_1")),
        reply(call("retrieve_clause", {"query": "focused reformulation"}, "search_2")),
        reply(call("check_requirement", {"retrieval_id": "retrieval_2"}, "check")),
        AIMessage(content="Done"),
    ])
    result = agent.run_agent("r1", "Original requirement", store, client=client, k=k)
    assert result.error is None and result.retrieved_clause_ids == ["4.1.2"]
    assert [(c.args[0], c.kwargs) for c in store.similarity_search.call_args_list] == [
        ("original query", {"k": k}), ("focused reformulation", {"k": k}),
    ]
    assert f"k={k}" in histories[0][0].content
    prompt = client.with_structured_output.return_value.invoke.call_args.args[0][1]["content"]
    assert "Original requirement" in prompt
    assert "focused reformulation" not in prompt


@pytest.mark.parametrize("k", [0, 11, True, "3", None, 1.5])
def test_invalid_k_fails_before_client_creation_or_model_work(setup, monkeypatch, k):
    store, client = setup
    create_client = Mock()
    monkeypatch.setattr(rag, "create_audit_client", create_client)
    with pytest.raises(ValueError, match="k must be an integer between 1 and 10"):
        agent.run_agent("r1", "Original", store, k=k)
    with pytest.raises(ValueError, match="k must be an integer between 1 and 10"):
        agent.AuditTools("r1", "Original", store, client, k=k)
    create_client.assert_not_called()
    client.bind_tools.assert_not_called()
    store.similarity_search.assert_not_called()


def test_default_client_reuses_canonical_configuration(setup, monkeypatch):
    store, client = setup
    create = Mock(return_value=client)
    monkeypatch.setattr(rag, "create_audit_client", create)
    script(client, workflow())
    assert agent.run_agent("r1", "Original", store).error is None
    create.assert_called_once_with()


def test_default_client_creation_failure_is_explicit(setup, monkeypatch):
    monkeypatch.setattr(rag, "create_audit_client", Mock(side_effect=RuntimeError("Configuration failed")))
    result = agent.run_agent("r1", "Original", setup[0])
    assert "Configuration failed" in result.error and result.verdict is None


@pytest.mark.parametrize("requirement_id,text,steps", [
    (" ", "Original", 8), ("r1", " ", 8), (None, "Original", 8),
    ("r1", "Original", 0), ("r1", "Original", True),
])
def test_invalid_run_inputs_fail_before_model_work(setup, requirement_id, text, steps):
    store, client = setup
    with pytest.raises(ValueError):
        agent.run_agent(requirement_id, text, store, client=client, max_steps=steps)
    client.bind_tools.assert_not_called()
