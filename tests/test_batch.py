"""Controls for the scoring job: the contract at the entrance, and what comes out.

    python -m pytest -q          # from the repository root

Everything else in this suite checks an analysis of data whose answers are known.
These check the other half -- scoring patients nobody has diagnosed yet, which is
the only thing a model is ultimately for, and the half where a mistake reaches a
person instead of a paragraph.

Two ideas are being tested. First, that the contract knows the difference between
a rule about the data and a rule about the cohort: a batch of forty new patients
is not a broken copy of wdbc.data. Second, that a batch is read by column NAME,
so the silent mislabelling that positional reading allows cannot happen at the
entrance to a scoring job.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import manifest, predict
from src.load_data import (
    FEATURES,
    MIN_SAMPLES_FOR_A_RATE,
    SMOOTH_OUTLINE_COLUMNS,
    load_data,
    read_batch,
    validate,
    validate_batch,
)
from src.model import BAND


@pytest.fixture(scope="module")
def cohort():
    """The real file, which is the only thing every batch here is carved from."""
    return load_data()


@pytest.fixture(scope="module")
def batch(cohort):
    """Forty patients as they would actually arrive: no diagnosis, and a header.

    Sampled rather than sliced. wdbc.data is not in random order -- five of its
    thirteen smooth-outline samples are in the last forty rows -- so `.tail(40)`
    is a batch from a different population, which the contract correctly refuses.
    """
    return cohort.drop(columns="diagnosis").sample(40, random_state=0)


@pytest.fixture
def written(tmp_path, monkeypatch):
    """Point the batch outputs at a throwaway directory, not the project's own."""
    monkeypatch.setattr(manifest, "RESULTS", tmp_path / "results")
    return tmp_path


def as_csv(frame, path, columns=None):
    """Write a batch the way one would arrive: a header row, and an id column."""
    frame[columns if columns is not None else list(frame.columns)].to_csv(path)
    return path


# --- the split: what is a rule about the data, and what about the cohort? ----

def test_a_batch_of_new_patients_satisfies_the_batch_contract(batch):
    """Forty patients, no diagnosis, and nothing wrong with them.

    This is the check the old contract could not express. Every rule it applied
    described wdbc.data itself, so the only data it could accept was wdbc.data.
    """
    assert validate_batch(batch) is batch


def test_the_same_batch_is_not_the_training_file(batch):
    """And the cohort rules still say so, loudly.

    The point of the split is not that the rules got weaker. 569 rows and 37%
    malignant are still required of the file the published numbers came from.
    """
    with pytest.raises(ValueError, match="expected 569 rows"):
        validate(batch)


def test_the_training_file_still_has_to_satisfy_both(cohort):
    """The contract that was there before the split still holds afterwards."""
    assert validate(cohort) is cohort


# --- the batch contract still refuses the things that matter -----------------

def test_a_batch_missing_a_measurement_is_refused(batch):
    """Thirty features in, or nothing. A model cannot be handed twenty-nine."""
    with pytest.raises(ValueError, match="measurements missing: texture_mean"):
        validate_batch(batch.drop(columns="texture_mean"))


def test_a_batch_carrying_an_unknown_column_is_refused(batch):
    """An extra column is evidence that this is not the file you think it is."""
    extra = batch.assign(scanner_serial=1)

    with pytest.raises(ValueError, match="unexpected columns: scanner_serial"):
        validate_batch(extra)


def test_a_batch_with_a_missing_value_is_refused(batch):
    """No sample in this dataset is missing a measurement. If one is, it is new."""
    broken = batch.copy()
    broken.loc[broken.index[0], "texture_mean"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_batch(broken)


def test_a_batch_of_negative_measurements_is_refused(batch):
    """Every one of the thirty numbers is a distance, an area or a ratio."""
    broken = batch.copy()
    broken.loc[broken.index[0], "area_worst"] = -1.0

    with pytest.raises(ValueError, match="negative values"):
        validate_batch(broken)


def test_an_empty_batch_is_refused(batch):
    """A job that scored nothing should say so rather than report a clean run."""
    with pytest.raises(ValueError, match="empty"):
        validate_batch(batch.head(0))


def test_a_tray_of_flat_outlines_is_still_refused(batch):
    """The failure that looks like real data, on a batch this time.

    A hundred samples with no concave points is an edge detector that stopped
    working, and every value it produced is still a plausible-looking number.
    """
    broken = pd.concat([batch] * 3)
    broken.index = range(len(broken))
    broken.loc[broken.index[:60], SMOOTH_OUTLINE_COLUMNS] = 0.0

    with pytest.raises(ValueError, match="edge detection"):
        validate_batch(broken)


def test_the_rate_rule_does_not_fire_on_a_handful_of_samples(batch):
    """A proportion needs enough samples to be a proportion.

    One smooth outline in a batch of five is 20%, and it is one patient with a
    small round nucleus. A check that fires on that is a check somebody switches
    off, which costs more than the rule was ever worth.
    """
    tiny = batch.head(5).copy()
    tiny.loc[tiny.index[0], SMOOTH_OUTLINE_COLUMNS] = 0.0

    assert len(tiny) < MIN_SAMPLES_FOR_A_RATE
    assert validate_batch(tiny) is tiny


# --- read by name, not by position ------------------------------------------

def test_a_batch_is_read_by_column_name_not_by_position(batch, tmp_path):
    """The failure the run manifest cannot catch, removed rather than guarded.

    wdbc.data has no header, so its columns are identified by position, and
    swapping two of them mislabels every downstream number without anything
    crashing. A checksum catches that for one fixed file. A batch is new every
    time and has no checksum to be compared against, so it carries a header and
    is read by name -- and the order it happens to arrive in stops mattering.
    """
    alphabetical = as_csv(batch, tmp_path / "batch.csv", columns=sorted(FEATURES))

    read = read_batch(alphabetical)

    assert list(read.columns) == FEATURES            # put back in the model's order
    pd.testing.assert_frame_equal(read, batch[FEATURES])


def test_reading_that_same_file_by_position_would_change_the_answers(batch, tmp_path):
    """The test that stops the one above being decorative.

    If column order did not matter, reading by name would be a stylistic choice.
    Scoring the same forty patients from the same file, read positionally,
    changes most of the diagnoses -- which is what it is buying.
    """
    path = as_csv(batch, tmp_path / "batch.csv", columns=sorted(FEATURES))
    model, _ = predict.load_model()

    correct, _ = predict.score(model, validate_batch(read_batch(path)))

    positional = pd.read_csv(path).set_index("id")
    positional.columns = FEATURES                    # the mislabelling, committed
    mislabelled = model.predict_proba(positional[FEATURES])[:, 1]

    assert correct == pytest.approx(model.predict_proba(batch[FEATURES])[:, 1])
    assert ((mislabelled >= 0.5) != (correct >= 0.5)).sum() > 20


# --- the job leaves a record -------------------------------------------------

def test_scoring_a_batch_writes_predictions_and_a_record(batch, tmp_path, written):
    """One row per sample, and one record naming everything that produced them."""
    path = as_csv(batch, tmp_path / "batch.csv", columns=sorted(FEATURES))

    assert predict.main([str(path)]) == 0

    predictions = sorted((written / "results").glob("batch-*.csv"))
    records = sorted((written / "results").glob("batch-*.json"))
    assert len(predictions) == 1 and len(records) == 1
    # The CSV and the JSON that describes it are named together, never separately.
    assert predictions[0].stem == records[0].stem

    scored = pd.read_csv(predictions[0]).set_index("id")
    assert len(scored) == len(batch)
    assert set(scored["decision"]) <= {"benign", "refer", "malignant"}

    record = json.loads(records[0].read_text())
    assert record["input"]["n_samples"] == len(batch)
    assert record["model"]["sha256"] == predict.load_model()[1]
    assert record["settings"]["referral_band"] == list(BAND)
    assert sum(record["outcome"]["counts"].values()) == len(batch)


def test_the_record_states_the_rates_that_can_be_watched_without_labels(batch, tmp_path, written):
    """Accuracy is months away. These two numbers are available on the day.

    They are the early warning: a large move in either says the incoming
    population or the measurements have changed, before any label arrives to
    confirm it.
    """
    path = as_csv(batch, tmp_path / "batch.csv", columns=sorted(FEATURES))
    predict.main([str(path)])

    record = json.loads(next((written / "results").glob("batch-*.json")).read_text())
    counts = record["outcome"]["counts"]

    assert record["outcome"]["positive_call_rate"] == pytest.approx(
        counts["malignant"] / len(batch), abs=0.0001
    )
    assert record["outcome"]["referral_rate"] == pytest.approx(
        counts["refer"] / len(batch), abs=0.0001
    )
    assert "accuracy" not in json.dumps(record["outcome"])


def test_a_batch_that_fails_the_contract_is_not_scored(batch, tmp_path, written):
    """Nothing written, nothing scored, and a non-zero exit.

    A scoring job that half-runs is worse than one that stops: the predictions it
    did write look exactly like predictions that were fine.
    """
    broken = batch.copy()
    broken.loc[broken.index[0], "radius_mean"] = -1.0
    path = as_csv(broken, tmp_path / "bad.csv", columns=sorted(FEATURES))

    assert predict.main([str(path)]) == 1
    assert not (written / "results").exists()


def test_a_batch_without_a_header_is_refused(tmp_path, written):
    """wdbc.data itself, handed to the scoring job, is not a batch.

    It is the likeliest wrong file to be pointed at, and it has no header row --
    so the first patient becomes the column names and everything after is
    mislabelled. The entrance says no rather than working it out.
    """
    assert predict.main(["data/wdbc.data"]) == 1
    assert not (written / "results").exists()
