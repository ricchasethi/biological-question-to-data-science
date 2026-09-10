"""Controls for the input: is this the file we think it is, and does it behave?

    python -m pytest -q          # from the repository root

A laboratory does not trust a reagent because it has the right label on the
tube. It runs a control. These are the same idea, applied to a data file: they
are the checks that would have caught the mistakes that do not announce
themselves -- a mislabelled column, a silently overwritten file, a batch of
images where the edge detector failed.
"""

import numpy as np
import pandas as pd
import pytest

from src.fetch_data import find_problems, read_checksums
from src.load_data import (
    COLUMNS,
    SMOOTH_OUTLINE_COLUMNS,
    load_data,
    validate,
)


@pytest.fixture(scope="module")
def df():
    """The real data, read once and shared by every test in this file."""
    return load_data()


# --- is this the right file? ------------------------------------------------

def test_the_raw_file_is_the_one_the_articles_used():
    """The published numbers came from one exact file. This proves it is still here.

    A file with the right name and different contents is the failure this guards
    against, and it is not hypothetical: the notebook cell this replaced
    re-downloaded and overwrote data/wdbc.data on every run.
    """
    assert find_problems(read_checksums()) == []


def test_the_columns_are_means_then_ses_then_worsts():
    """The quietest way to be completely wrong.

    wdbc.data has no header row, so this list IS the column names. The file
    stores all ten means, then all ten standard errors, then all ten "worst"
    values. Build the names in the other order and every one of the thirty
    columns is mislabelled -- nothing crashes, the model still trains, and every
    result is nonsense.
    """
    assert len(COLUMNS) == 32
    assert COLUMNS[2] == "radius_mean"
    assert COLUMNS[11] == "fractal_dimension_mean"
    assert COLUMNS[12] == "radius_se"
    assert COLUMNS[22] == "radius_worst"


# --- does the contract accept good data? ------------------------------------

def test_the_contract_accepts_the_real_data(df):
    """The first thing to check about any test: that it passes on the real thing.

    A contract that rejects the dataset it was written for is not strict, it is
    broken.
    """
    assert validate(df) is df


def test_zero_concavity_is_a_reading_not_a_gap(df):
    """Thirteen benign samples record exactly zero concavity, and that is correct.

    A generic rule would call these missing values and drop or impute them.
    The biology says otherwise: a small, smooth nucleus has no detectable
    concave region, so zero is what the edge detector should report. They are
    zero in all six concavity columns at once, never in only some -- and they
    are the only zeros anywhere in the thirty measurements.
    """
    no_concave_points = (df[SMOOTH_OUTLINE_COLUMNS] == 0).all(axis=1)

    assert no_concave_points.sum() == 13
    assert set(df.loc[no_concave_points, "diagnosis"]) == {"B"}

    validate(df)   # and the contract lets them through


# --- does the contract reject bad data? -------------------------------------

def test_the_contract_rejects_a_truncated_file(df):
    """Half a download is the most ordinary way a data file goes wrong."""
    with pytest.raises(ValueError, match="expected 569 rows"):
        validate(df.head(100))


def test_the_contract_rejects_a_missing_value(df):
    """No sample in this dataset is missing a measurement. If one is, it is new."""
    broken = df.copy()
    broken.loc[broken.index[0], "texture_mean"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate(broken)


def test_the_contract_rejects_a_negative_measurement(df):
    """Every one of the thirty numbers is a distance, an area or a ratio."""
    broken = df.copy()
    broken.loc[broken.index[0], "texture_mean"] = -1.0

    with pytest.raises(ValueError, match="negative values"):
        validate(broken)


def test_the_contract_rejects_a_duplicated_patient(df):
    """One patient counted twice quietly weights the model towards that patient."""
    with pytest.raises(ValueError, match="appear twice"):
        validate(pd.concat([df, df.head(1)]))


def test_the_contract_rejects_a_tray_of_flat_outlines(df):
    """The failure that looks like real data.

    Thirteen samples with no concave points is biology. Two hundred is an edge
    detector that stopped working -- and every value it produced would still be
    a plausible-looking number in a well-formed file.
    """
    broken = df.copy()
    broken.loc[broken.index[:200], SMOOTH_OUTLINE_COLUMNS] = 0.0

    with pytest.raises(ValueError, match="edge detection"):
        validate(broken)


def test_the_contract_rejects_a_batch_from_a_different_population(df):
    """The model was validated at 37% malignant. A screening clinic is not.

    The data can be perfectly clean and still be the wrong data: accuracy
    measured on one mix of patients does not carry over to another.
    """
    mostly_benign = pd.concat([df[df["diagnosis"] == "B"], df[df["diagnosis"] == "M"].head(10)])
    padded = pd.concat([mostly_benign] * 2).head(569)
    padded.index = range(569)

    with pytest.raises(ValueError, match="malignant"):
        validate(padded)
