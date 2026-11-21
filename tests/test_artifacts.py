"""The committed artifacts must agree with each other.

A benchmark repository is only useful if the numbers in the README, the JSON
files and the shipped weights describe the same run. These tests fail when they
drift apart, which is the failure mode that is easiest to miss by eye.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from physics_surrogates.models.fno import evaluate, load_fno_weights, random_periodic_fields
from physics_surrogates.models.pinns import BurgersPINN
from physics_surrogates.reference import cole_hopf_burgers

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks"
WEIGHTS = ROOT / "weights"


def _results(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_readme_figures_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    references = sorted(set(re.findall(r"!\[[^\]]*\]\((docs/img/[^)\s]+)\)", readme)))
    assert references, "README references no figures"
    missing = [ref for ref in references if not (ROOT / ref).exists()]
    assert not missing, f"missing figures: {missing}"


@pytest.mark.parametrize("name", ["burgers_1d_results.json", "fno_heat_results.json", "mms_cavity_results.json"])
def test_result_files_are_present_and_valid_json(name):
    data = _results(name)
    assert isinstance(data, dict) and data


def test_burgers_json_describes_the_recorded_configuration():
    data = _results("burgers_1d_results.json")
    for key in ("nu", "training", "model", "system", "metrics", "validation", "weights_file"):
        assert key in data, key

    training = data["training"]
    assert training["n_ic"] > 0 and training["n_bc"] > 0
    assert training["n_colloc_final"] >= training["n_colloc"]
    assert data["model"]["params"] == 12737

    validation = data["validation"]
    assert validation["n_test_points"] == 200
    assert set(validation["by_time"]) == {"0.25", "0.50", "0.75", "1.00"}
    # The published headline number is the t = 0.5 entry, not a separate value.
    assert validation["l2_relative_error"] == validation["by_time"]["0.50"]["l2_relative_error"]


def test_burgers_weights_reproduce_the_recorded_metrics():
    """Load the shipped weights and recompute the published error."""
    data = _results("burgers_1d_results.json")
    pinn = BurgersPINN(nu=data["nu"], device="cpu")
    pinn.load_weights(ROOT / data["weights_file"])

    x = np.linspace(-1.0, 1.0, data["validation"]["n_test_points"])
    t = data["validation"]["t"]
    pred = pinn.predict(x, t)
    ref = cole_hopf_burgers(x, t, data["nu"])
    rel_l2 = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    assert rel_l2 == pytest.approx(data["validation"]["l2_relative_error"], rel=1e-6)


def test_burgers_error_grows_towards_t_equals_one():
    """The README claims this; the JSON must support it."""
    by_time = _results("burgers_1d_results.json")["validation"]["by_time"]
    ordered = [by_time[key]["l2_relative_error"] for key in sorted(by_time)]
    assert ordered[-1] > ordered[0]


def test_fno_json_covers_the_reported_grids():
    data = _results("fno_heat_results.json")
    assert set(data["test_rel_l2_by_grid"]) == {"64", "128", "256"}
    assert data["params"] == 74209
    assert "system" in data
    values = data["test_rel_l2_by_grid"].values()
    assert all(0 < value < 1e-2 for value in values)


def test_fno_weights_reproduce_the_recorded_errors():
    data = _results("fno_heat_results.json")
    model = load_fno_weights(WEIGHTS / "fno_heat.pt")
    for grid, expected in data["test_rel_l2_by_grid"].items():
        fields = random_periodic_fields(200, int(grid), seed=1234)
        measured = evaluate(model, fields, data["nu"], data["T"])
        assert measured == pytest.approx(expected, rel=1e-3, abs=1e-9), grid
