"""Behavioral checks for the modular pipeline and local notebook conversion."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.compliance_rag.data import load_clauses, load_benchmark, discover_csvs, combine_benchmarks
from src.compliance_rag.embeddings import TfidfEmbedder
from src.compliance_rag.retrieval import ClauseIndex
from src.compliance_rag.judges import HeuristicJudge, OllamaJudge
from src.compliance_rag.pipeline import run_pipeline
from src.compliance_rag.evaluation import evaluate
from src.compliance_rag.schemas import AuditVerdict

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_entrypoint_reports_and_resume(tmp_path):
    script = load_script('compliance_rag_baseline')
    args = ['--clauses', str(ROOT / 'data/npr-7150-2d-requirements.csv'),
            '--benchmark', str(ROOT / 'data/npr7150_ground_truth_benchmark.csv'),
            '--embedder', 'tfidf', '--judge', 'heuristic', '--limit', '5', '--out', str(tmp_path)]
    frame, metrics = script.main(args)
    with patch.object(HeuristicJudge, 'judge', side_effect=AssertionError('cache missed')):
        resumed, resumed_metrics = script.main(args)
    pd.testing.assert_frame_equal(frame, resumed)
    assert metrics == resumed_metrics
    assert len((tmp_path / 'cache.jsonl').read_text().splitlines()) == 5
    assert len(json.loads((tmp_path / 'audit_records.json').read_text())) == 5
    assert '# October Milestone' in (tmp_path / 'report.md').read_text()
    assert json.loads((tmp_path / 'metrics.json').read_text())['n'] == 5


def test_invalid_grounding_is_retried_and_not_cached(tmp_path):
    clauses = load_clauses(ROOT / 'data/npr-7150-2d-requirements.csv')
    bench = load_benchmark(ROOT / 'data/npr7150_ground_truth_benchmark.csv').head(1)
    index = ClauseIndex(clauses, TfidfEmbedder([c.text for c in clauses]))
    payload = dict(clause_id='999.999', evidence='test', missing_elements=['test'],
                   rationale='The retrieved obligation is missing.', verdict='Gap')
    judge = OllamaJudge(max_retries=1)
    judge.client = Mock()
    judge.client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    cache = tmp_path / 'nested/cache.jsonl'
    records = run_pipeline(bench, index, judge, 5, str(cache))
    assert judge.client.chat.call_count == 2
    assert records[0].verdict is None
    assert not records[0].grounded
    assert not cache.exists()
    _, metrics, _ = evaluate(bench, records, 5)
    assert metrics['schema_valid_rate'] == 0


def test_schema_consistency():
    with pytest.raises(ValueError, match='missing_elements'):
        AuditVerdict(clause_id='1', evidence='test', missing_elements=['missing'],
                     rationale='This rationale is long enough.', verdict='Meets')


def test_local_notebook_conversion_combines_data_and_archives(tmp_path):
    script = load_script('compliance_copilot_october_ollama')
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    pd.DataFrame([dict(clause='1', swe_id='SWE-1', requirement='Must test')]).to_csv(data_dir / 'clauses.csv', index=False)
    row = dict(requirement_text='We test', clause_id='1', label='Meets')
    pd.DataFrame([row, row]).to_csv(data_dir / 'benchmark.csv', index=False)
    clauses, benchmarks = discover_csvs(data_dir)
    assert clauses.name == 'clauses.csv'
    assert len(combine_benchmarks(benchmarks)) == 1
    out = tmp_path / 'run'

    def baseline(args):
        (out / 'report.md').write_text('Report')
        assert '--limit' not in args  # 0 requests all rows
        assert args[args.index('--model') + 1] == 'test-model'
        return pd.DataFrame(), {'n': 1}

    with patch('ollama.Client') as client, patch.object(script, 'run_baseline', side_effect=baseline):
        script.main(['--data-dir', str(data_dir), '--out', str(out), '--limit', '0',
                     '--model', 'test-model', '--archive'])
        client.return_value.list.assert_called_once()
    assert pd.read_csv(out / 'combined_benchmark.csv')['requirement_id'].tolist() == [1]
    assert out.with_suffix('.zip').exists()


def test_nonpositive_retrieval_k_rejected(tmp_path):
    script = load_script('compliance_rag_baseline')
    with pytest.raises(SystemExit):
        script.main(['--clauses', 'unused', '--benchmark', 'unused', '--k', '0'])
