"""Result files say where a run executed, and whether it shared its GPU.

A run that shared its GPU with others (`scripts/m3_job.sh --parallel`) timed
QUANT three times slower than one alone, so its timings are not comparable with
anything else. What is worth pinning is that the marker reaches every model
entry when it applies, and that laptop runs, outside Slurm, write exactly what
they wrote before.
"""

import json

import pytest

from fit2082.results import run_context, write_results


@pytest.fixture
def outside_slurm(monkeypatch):

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("FIT2082_GPU_RUNS", raising=False)


def test_no_context_outside_slurm(outside_slurm, tmp_path):

    assert run_context() == {}

    out = write_results(
        tmp_path / "r.json", {"a": {"x_name": "round"}}, {"commit": "c"}
    )

    assert json.loads(out.read_text())["models"] == {"a": {"x_name": "round"}}


def test_a_job_alone_is_not_shared(outside_slurm, monkeypatch):

    monkeypatch.setenv("SLURM_JOB_ID", "123")

    context = run_context()

    assert context["slurm_job"] == "123"
    assert context["gpu_runs"] == 1
    assert context["shared_gpu"] is False


def test_every_entry_of_a_shared_run_is_marked(outside_slurm, monkeypatch, tmp_path):

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("FIT2082_GPU_RUNS", "3")

    out = write_results(
        tmp_path / "r.json", {"a": {"x_name": "round"}, "b": {}}, {"commit": "c"}
    )
    models = json.loads(out.read_text())["models"]

    for entry in models.values():
        assert entry["context"]["gpu_runs"] == 3
        assert entry["context"]["shared_gpu"] is True
