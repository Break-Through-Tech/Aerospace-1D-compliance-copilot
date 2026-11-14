"""Canonical corpus integrity and disposable Chroma rebuilds, without Ollama."""

import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from scripts import build_index
from src import vector_store
from src.config import DATA_PATH
from src.preprocessing import create_documents, load_requirements
from src.schemas import Clause

DATA = Path(__file__).resolve().parents[1] / "data"


def test_canonical_corpus_preserves_reviewed_text_ids_and_metadata():
    original = json.loads((DATA / "npr-7150-2d-requirements.json").read_text())
    fixed = json.loads((DATA / "npr-7150-2d-requirements_fixed.json").read_text())
    assert DATA_PATH == DATA / "npr-7150-2d-requirements_fixed.csv"
    with DATA_PATH.open(newline="") as source:
        assert list(csv.DictReader(source)) == fixed
    assert len(original) == len(fixed) == 130
    assert len({row["swe_id"] for row in fixed}) == len({row["clause"] for row in fixed}) == 130
    assert sum(before != after for before, after in zip(original, fixed)) == 62
    for before, after in zip(original, fixed):
        assert {k: v for k, v in before.items() if k != "requirement"} == {
            k: v for k, v in after.items() if k != "requirement"
        }
        assert after["requirement"].startswith(before["requirement"])
    # Pin reviewed bytes, including list tails, notes, indentation and table cells.
    hashes = {
        "npr-7150-2d-requirements.csv": "bbea7241d65865048a318e61720e309cfd5e2e007e1a4be2c9e63a6e2b04cf96",
        "npr-7150-2d-requirements.json": "b0f36edeade342354f3955c1d546392ec06ac5b17398dc10c22a9845cb3ea3d7",
        "npr-7150-2d-requirements_fixed.csv": "48cd6e25faeefe0051d236372809aaf9eddd0c64b7e629235bb388461c01f938",
        "npr-7150-2d-requirements_fixed.json": "00c5a8c558eac2802a002335429dadc7d1cc813d71a97751ea6b7c1ca3ccfab7",
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
    documents = create_documents(load_requirements(DATA_PATH))
    for row, doc in zip(fixed, documents):
        assert doc.id == row["swe_id"]
        assert doc.page_content == doc.metadata["section_title"] + "\n\n" + row["requirement"]
        assert doc.metadata == {
            "swe_id": row["swe_id"], "clause": row["clause"],
            "requirement": row["requirement"], "source": row["source_document"],
            "source_url": row["source_url"],
            "chapter_title": doc.metadata["chapter_title"],
            "section_title": doc.metadata["section_title"],
        }
        Clause.model_validate(doc.metadata)


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.calls = []
        self.fail = False

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("Embedding service unavailable")
        self.calls.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    embeddings = LocalEmbeddings()
    monkeypatch.setattr(vector_store, "VECTOR_STORE_PATH", tmp_path / "chroma")
    monkeypatch.setattr(vector_store, "COLLECTION_NAME", "nasa_test")
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(build_index, "VECTOR_STORE_PATH", vector_store.VECTOR_STORE_PATH)
    monkeypatch.setattr(build_index, "COLLECTION_NAME", vector_store.COLLECTION_NAME)
    return embeddings


@pytest.mark.parametrize("existing_empty", [False, True])
@pytest.mark.parametrize("args", [[], ["--rebuild"]])
def test_cli_builds_absent_or_empty_collection(isolated_store, capsys, existing_empty, args):
    if existing_empty:
        vector_store.Chroma(
            collection_name=vector_store.COLLECTION_NAME,
            persist_directory=str(vector_store.VECTOR_STORE_PATH),
            embedding_function=isolated_store,
        )
    build_index.main(args)
    stored = vector_store.load_vector_store().get(include=["documents", "metadatas"])
    documents = create_documents(load_requirements(DATA_PATH))
    assert len(stored["ids"]) == 130
    assert dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"]))) == {
        doc.id: (doc.page_content, doc.metadata) for doc in documents
    }
    assert "Verified 130 indexed records" in capsys.readouterr().out


def test_cli_rebuild_removes_stale_records_reembeds_and_preserves_other_collections(isolated_store, capsys):
    original = create_documents(load_requirements(DATA / "npr-7150-2d-requirements.csv"))
    store = vector_store.build_vector_store(original)
    store.add_documents([Document(id="stale", page_content="Outdated record")])
    other = vector_store.Chroma(
        collection_name="unrelated", persist_directory=str(vector_store.VECTOR_STORE_PATH),
        embedding_function=isolated_store,
    )
    other.add_documents([Document(id="keep", page_content="Unrelated record")])
    before = store.get(include=["documents", "metadatas"])
    calls_before = list(isolated_store.calls)
    with pytest.raises(ValueError, match="explicit rebuild"):
        build_index.main([])
    assert store.get(include=["documents", "metadatas"]) == before
    assert isolated_store.calls == calls_before
    assert "Verified" not in capsys.readouterr().out
    fixed = create_documents(load_requirements(DATA_PATH))
    for _ in range(2):
        isolated_store.calls.clear()
        build_index.main(["--rebuild"])
        store = vector_store.load_vector_store()
        assert isolated_store.calls == [doc.page_content for doc in fixed]
        stored = store.get(include=["documents", "metadatas", "embeddings"])
        assert len(stored["ids"]) == 130
        assert dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"]))) == {
            doc.id: (doc.page_content, doc.metadata) for doc in fixed
        }
        assert len(stored["embeddings"]) == 130
        assert other.get()["ids"] == ["keep"]
        assert "Verified 130 indexed records" in capsys.readouterr().out


def test_failed_rebuild_propagates_and_empty_index_cannot_be_loaded(isolated_store, capsys):
    documents = create_documents(load_requirements(DATA_PATH))
    vector_store.build_vector_store(documents)
    isolated_store.fail = True
    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        build_index.main(["--rebuild"])
    assert "Verified" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="empty"):
        vector_store.load_vector_store()


@pytest.mark.parametrize("documents", [[], [Document(page_content="Missing ID")], [
    Document(id="same", page_content="First"), Document(id="same", page_content="Second"),
]])
def test_invalid_ids_fail_before_opening_chroma(monkeypatch, documents):
    factory = Mock()
    monkeypatch.setattr(vector_store, "Chroma", factory)
    with pytest.raises(ValueError, match="unique IDs"):
        vector_store.build_vector_store(documents, rebuild=True)
    factory.assert_not_called()


@pytest.mark.parametrize("args", [[], ["--rebuild"]])
def test_cli_verification_failure_never_reports_success(monkeypatch, capsys, args):
    build = Mock(side_effect=ValueError("Index verification failed"))
    monkeypatch.setattr(build_index, "build_vector_store", build)
    with pytest.raises(ValueError, match="verification failed"):
        build_index.main(args)
    assert build.call_args.kwargs == {"rebuild": bool(args)}
    assert "Verified" not in capsys.readouterr().out


def test_inexact_readback_fails_verification(monkeypatch):
    store = Mock()
    store.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    monkeypatch.setattr(vector_store, "Chroma", Mock(return_value=store))
    with pytest.raises(ValueError, match="verification failed"):
        vector_store.build_vector_store([Document(id="one", page_content="Full text")], rebuild=True)
