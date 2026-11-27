"""Tests for GWLab Phase 2 read-only data layer, schemas, and loaders."""

import math
from pathlib import Path
import numpy as np
import pytest

from gwlab.data.provenance import (
    ProvenanceCategory,
    ArtifactValidationStatus,
    ProvenanceRecord,
)
from gwlab.data.schemas import (
    RunResult,
    AggregateLedger,
    PosteriorSamples,
    SeedComparisonResult,
)
from gwlab.data.loaders import (
    resolve_safe_artifact_path,
    load_stage6c_result,
    load_aggregate_ledger,
    load_posterior_samples,
    load_seed_comparison_pair,
)


def test_resolve_safe_artifact_path_valid():
    """Verify safe path resolution against repository root."""
    path = resolve_safe_artifact_path("experiments/wp6_stage6c_anchor/run_a_anchor_result.json")
    assert path.is_file()
    assert path.name == "run_a_anchor_result.json"


def test_resolve_safe_artifact_path_rejections():
    """Verify rejection of empty, absolute, or escaping paths."""
    with pytest.raises(ValueError, match="Artifact path must be a non-empty string"):
        resolve_safe_artifact_path("")

    with pytest.raises(ValueError, match="Absolute paths are prohibited"):
        resolve_safe_artifact_path(str(Path("/escaped.json").resolve()))

    with pytest.raises(ValueError, match="escapes repository root"):
        resolve_safe_artifact_path("../../outside_repo.json")

    with pytest.raises(FileNotFoundError, match="Scientific artifact not found"):
        resolve_safe_artifact_path("experiments/non_existent_file.json")


def test_load_stage6c_result_authoritative_values():
    """Verify loading of Stage 6C Run A result with exact evidence and call count."""
    result = load_stage6c_result()

    assert isinstance(result, RunResult)
    assert result.run_id == "run_a_anchor"
    assert result.status == "SUCCESS"

    # Authoritative evidence verification
    assert math.isclose(result.log_evidence, 201646.1413147189, rel_tol=1e-12)
    assert math.isclose(result.log_evidence_err, 0.23736832901350433, rel_tol=1e-12)

    # Reconciled evaluation count
    assert result.authoritative_call_count == 487407
    assert result.n_evaluations == 500657

    # Runtime verification
    assert result.wall_time_s > 0.0
    assert result.cpu_time_s > 0.0

    # Parameters summary verification
    assert set(result.parameters.keys()) == {"mc", "eta", "tc", "phic"}
    assert math.isclose(result.parameters["mc"].median, 28.196767, rel_tol=1e-4)
    assert math.isclose(result.parameters["eta"].median, 0.248101, rel_tol=1e-4)

    # Provenance
    assert result.provenance.category == ProvenanceCategory.SAVED_POSTERIOR
    assert result.provenance.validation_status == ArtifactValidationStatus.VALID
    assert result.provenance.seed == 8101


def test_load_aggregate_ledger_authoritative_values():
    """Verify loading of computational resource ledger and remaining budgets."""
    ledger = load_aggregate_ledger()

    assert isinstance(ledger, AggregateLedger)
    assert ledger.cumulative_calls == 487407
    assert ledger.remaining_calls == 712593
    assert ledger.total_budget_calls == 1200000

    assert math.isclose(ledger.cumulative_cpu_s, 473.15625, rel_tol=1e-5)
    assert math.isclose(ledger.remaining_cpu_s, 1326.84375, rel_tol=1e-5)
    assert math.isclose(ledger.total_budget_cpu_s, 1800.0, rel_tol=1e-5)

    assert math.isclose(ledger.cumulative_wall_s, 514.9467866, rel_tol=1e-5)
    assert math.isclose(ledger.remaining_wall_s, 2485.0532134, rel_tol=1e-5)
    assert math.isclose(ledger.total_budget_wall_s, 3000.0, rel_tol=1e-5)

    assert ledger.provenance.category == ProvenanceCategory.DERIVED_STATISTIC
    assert ledger.provenance.validation_status == ArtifactValidationStatus.VALID


def test_load_posterior_samples_authoritative_arrays():
    """Verify NPZ loading with allow_pickle=False, shape checks, and weight sum."""
    samples_obj = load_posterior_samples()

    assert isinstance(samples_obj, PosteriorSamples)
    assert samples_obj.run_id == "run_a_anchor"
    assert samples_obj.n_samples == 11951
    assert samples_obj.n_params == 4
    assert samples_obj.parameter_names == ["mc", "eta", "tc", "phic"]

    # Strict weight normalization check
    weights_sum = float(np.sum(samples_obj.weights))
    assert math.isclose(weights_sum, 1.0, abs_tol=1e-4)

    # Finiteness check
    assert np.all(np.isfinite(samples_obj.samples))
    assert np.all(np.isfinite(samples_obj.weights))

    assert samples_obj.provenance.category == ProvenanceCategory.SAVED_POSTERIOR
    assert samples_obj.provenance.validation_status == ArtifactValidationStatus.VALID


def test_load_seed_comparison_pair_authoritative_values():
    """Verify historical two-seed repeatability comparison calculation."""
    comp = load_seed_comparison_pair()

    assert isinstance(comp, SeedComparisonResult)
    assert comp.seed_1 == 8101
    assert comp.seed_2 == 8201

    assert math.isclose(comp.logz_1, 201646.1413147189, rel_tol=1e-12)
    assert math.isclose(comp.logz_2, 201643.1692960849, rel_tol=1e-12)

    assert math.isclose(comp.delta_logz, 2.972018634, rel_tol=1e-4)
    assert math.isclose(comp.sigma_delta, 0.33320958, rel_tol=1e-4)
    assert math.isclose(comp.z_discrepancy, 8.919367, rel_tol=1e-2)

    # Scientific qualification text check
    assert "not a calibrated frequentist significance" in comp.qualification
    assert "frozen inference problem" in comp.qualification


def test_loader_rejects_malformed_json(tmp_path, monkeypatch):
    """Verify loader raises clear ValueError on corrupt JSON files."""
    import gwlab.data.loaders as loaders

    # Create temporary corrupt JSON
    corrupt_file = tmp_path / "corrupt_result.json"
    corrupt_file.write_text("{ incomplete json: ", encoding="utf-8")

    # Temporarily monkeypatch WORKSPACE_ROOT
    monkeypatch.setattr(loaders, "WORKSPACE_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Corrupt JSON artifact"):
        loaders.load_stage6c_result("corrupt_result.json")


def test_loader_rejects_missing_json_keys(tmp_path, monkeypatch):
    """Verify loader raises KeyError on missing mandatory JSON fields."""
    import gwlab.data.loaders as loaders

    incomplete_file = tmp_path / "incomplete_result.json"
    incomplete_file.write_text('{"run_id": "test"}', encoding="utf-8")

    monkeypatch.setattr(loaders, "WORKSPACE_ROOT", tmp_path)

    with pytest.raises(KeyError, match="Missing mandatory keys"):
        loaders.load_stage6c_result("incomplete_result.json")


def test_loader_rejects_malformed_weights(tmp_path, monkeypatch):
    """Verify loader rejects NPZ where weights do not sum to unity (no silent normalization)."""
    import gwlab.data.loaders as loaders

    bad_npz = tmp_path / "bad_weights.npz"
    np.savez(
        bad_npz,
        samples=np.ones((100, 4)),
        weights=np.ones(100),  # Sum is 100.0, not 1.0!
    )

    monkeypatch.setattr(loaders, "WORKSPACE_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Sample weights do not sum to unity"):
        loaders.load_posterior_samples("bad_weights.npz")


def test_loader_rejects_non_finite_arrays(tmp_path, monkeypatch):
    """Verify loader rejects NPZ containing NaN or Inf values."""
    import gwlab.data.loaders as loaders

    nan_npz = tmp_path / "nan_samples.npz"
    samples = np.ones((100, 4))
    samples[10, 1] = np.nan
    weights = np.full(100, 0.01)

    np.savez(nan_npz, samples=samples, weights=weights)
    monkeypatch.setattr(loaders, "WORKSPACE_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Non-finite values"):
        loaders.load_posterior_samples("nan_samples.npz")
