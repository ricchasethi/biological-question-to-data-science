"""Controls for the record: do the control patients work, and is the run recorded?

    python -m pytest -q          # from the repository root

Two things are being checked here. First, that the control patients would
actually notice a change -- a control that passes no matter what is worse than no
control, because it is reassuring. Second, that the manifest names everything a
future reader would need to reproduce a number.
"""

import json

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import manifest
from src.controls import CONTROLS, controls_that_moved, read_controls
from src.evaluate import held_out_results
from src.fetch_data import read_checksums
from src.load_data import DATA_FILE, features_and_labels, load_data, validate
from src.model import build_model, split_data


@pytest.fixture(scope="module")
def fitted():
    """The data, the split, and the model fitted exactly as the analysis fits it."""
    X, y = features_and_labels(validate(load_data()))
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model().fit(X_train, y_train)
    return X, X_test, y_test, model


@pytest.fixture
def somewhere_else(tmp_path, monkeypatch):
    """Point the manifest at a throwaway directory.

    A test should not write into the project's own results/ and models/. These are
    the same two paths the real run uses, redirected for the length of one test.
    """
    monkeypatch.setattr(manifest, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(manifest, "MODELS", tmp_path / "models")
    monkeypatch.setattr(manifest, "MODEL_FILE", tmp_path / "models" / "model.joblib")
    return tmp_path


# --- the control patients ---------------------------------------------------

def test_the_control_patients_read_what_they_should(fitted):
    """All three, to six decimal places, on the model the analysis builds."""
    X, _, _, model = fitted

    readings = read_controls(model, X)

    assert controls_that_moved(readings) == []
    assert set(readings) == set(CONTROLS)


def test_a_changed_model_moves_the_controls(fitted):
    """The test that stops the controls being decorative.

    Change the regularisation -- an ordinary, well-intentioned edit that leaves
    accuracy looking respectable -- and all three controls move. That is what a
    control is for: it notices the change that the headline number hides.
    """
    X, _, _, _ = fitted
    X_train, _, y_train, _ = split_data(*features_and_labels(validate(load_data())))
    tuned = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)
    ).fit(X_train, y_train)

    readings = read_controls(tuned, X)

    assert controls_that_moved(readings) == list(CONTROLS)


def test_the_borderline_control_is_the_cancer_we_miss():
    """879523 sits 0.008 below the threshold, which is why it is worth watching.

    A control at 0.0 and a control at 1.0 will survive almost any change. The one
    near the decision boundary is where drift shows up first.
    """
    assert CONTROLS[879523]["expected"] == pytest.approx(0.491610)
    assert CONTROLS[879523]["expected"] < 0.5


# --- the manifest -----------------------------------------------------------

def test_the_manifest_names_the_input_by_checksum(fitted, somewhere_else):
    """The checksum in the manifest must be the one in data/checksums.sha256.

    Two records of the same file's identity that can disagree are not two
    records, they are a bug waiting to happen.
    """
    X, X_test, y_test, model = fitted
    written = manifest.build_manifest(
        model, _cv_scores(), held_out_results(model, X_test, y_test), read_controls(model, X)
    )

    expected = {str(path): digest for path, digest in read_checksums().items()}
    assert written["input"]["sha256"] == expected[DATA_FILE]


def test_the_manifest_names_the_model_by_checksum(fitted, somewhere_else):
    """The same model, fitted twice, must produce the same checksum.

    If it did not, the model hash would record nothing except the time of day.
    """
    X, X_test, y_test, model = fitted
    results, controls = held_out_results(model, X_test, y_test), read_controls(model, X)

    first = manifest.build_manifest(model, _cv_scores(), results, controls)
    second = manifest.build_manifest(model, _cv_scores(), results, controls)

    assert first["model"]["sha256"] == second["model"]["sha256"]
    assert manifest.MODEL_FILE.exists()


def test_the_manifest_records_whether_the_code_was_committed():
    """A commit hash with uncommitted edits on top is a claim you cannot support."""
    code = manifest.code_version()

    assert set(code) >= {"commit", "uncommitted_changes"}
    if code["commit"] is not None:
        assert len(code["commit"]) == 40
        assert isinstance(code["uncommitted_changes"], bool)


def test_a_run_writes_one_manifest_that_reads_back(fitted, somewhere_else):
    """It has to be a real file, and it has to still be JSON on the way back in."""
    X, X_test, y_test, model = fitted

    path = manifest.write_manifest(
        model, _cv_scores(), held_out_results(model, X_test, y_test), read_controls(model, X)
    )

    assert path.parent == manifest.RESULTS
    assert path.name.startswith("run-")

    written = json.loads(path.read_text())
    assert written["results"]["false_negatives"] == 3
    assert written["controls"]["readings"]["879523"]["passed"] is True
    assert all(written["environment"]["packages"].values())


def _cv_scores():
    """Stand-in for the cross-validation scores, so these tests stay fast.

    The manifest only reads .mean() and .std() off them, and whether the numbers
    are right is already tested in test_analysis.py.
    """
    import numpy as np
    return np.array([0.97, 0.98, 0.97, 0.96, 0.98])
