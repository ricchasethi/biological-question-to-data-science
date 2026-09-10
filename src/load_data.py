"""Turn the raw file into a table the analysis can use.

    from src.load_data import load_data, validate, features_and_labels

    df = validate(load_data())
    X, y = features_and_labels(df)

load_data() reads the file. validate() is the data contract: it checks the table
against everything the analysis assumes and refuses to hand back data the model
was not built for. They are separate so you can look at a file that fails.

wdbc.data has NO header row. The column names live here, in one place, because
getting them wrong is the quietest way to be completely wrong: every column
would be mislabelled, nothing would crash, and the model would still train.
"""

import pandas as pd

# The ten measurements taken of each cell nucleus, in the order the file stores them.
MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]

# Each measurement is summarised three ways across the nuclei in one image.
SUMMARIES = ["mean", "se", "worst"]

# The order of these two loops is the whole game. The file stores all ten means
# first, then all ten standard errors, then all ten "worst" values. Swap the loops
# and every one of the 30 columns gets the wrong name.
COLUMNS = ["id", "diagnosis"] + [
    f"{measurement}_{summary}"
    for summary in SUMMARIES
    for measurement in MEASUREMENTS
]

# The ten "mean" columns, used wherever a figure or a table shows one summary only.
MEAN_FEATURES = [f"{measurement}_mean" for measurement in MEASUREMENTS]

DATA_FILE = "data/wdbc.data"


def load_data(path=DATA_FILE):
    """Read wdbc.data into a table of 569 rows, indexed by patient id."""
    return pd.read_csv(path, header=None, names=COLUMNS).set_index("id")


def features_and_labels(df):
    """Split the table into what the model sees and what it has to predict.

    X = the 30 measurements.
    y = 1 for malignant, 0 for benign. Malignant is 1 because it is the thing
        we are trying to find, and that is what recall and precision count.
    """
    X = df.drop(columns="diagnosis")
    y = (df["diagnosis"] == "M").astype(int)
    return X, y


# ---------------------------------------------------------------------------
# The data contract
# ---------------------------------------------------------------------------
# Everything below is a statement about the data that the analysis takes for
# granted. Writing them down turns an assumption into a check: if a future file
# breaks one, the run stops instead of quietly producing a plausible number.

# The dataset is fixed and complete. A different row count means a different file.
N_SAMPLES = 569

# 37.3% of these samples are malignant. A batch that arrives at 5% or 80% may be
# perfectly good data from a different population -- but it is not the population
# this model was validated on, and its accuracy would not carry across.
MALIGNANT_FRACTION = (0.30, 0.45)

# The six columns where zero is a real measurement, not a gap.
# A small, smooth nucleus has no detectable concave region, so the edge detector
# correctly reports 0.0. Thirteen samples do this, all benign, and they are zero
# in all six columns at once -- never in only some of them.
SMOOTH_OUTLINE_COLUMNS = [
    "concavity_mean", "concave_points_mean",
    "concavity_se", "concave_points_se",
    "concavity_worst", "concave_points_worst",
]

# 13/569 = 2.3% of samples have no concave points. If a batch arrives where a
# tenth of the images say that, the likelier explanation is a failed edge
# detector than a tray full of unusually smooth nuclei.
MAX_SMOOTH_OUTLINES = 0.10


def validate(df):
    """Check the table against everything the analysis assumes, before using it.

    Returns the table unchanged, so it can be used in place:

        df = validate(load_data())

    Raises ValueError listing *every* rule that failed, not just the first one.
    A contract that reports one problem at a time makes you fix a bad file in
    ten runs instead of one.
    """
    problems = []

    # --- shape -------------------------------------------------------------
    if len(df) != N_SAMPLES:
        problems.append(f"expected {N_SAMPLES} rows, found {len(df)}")

    if list(df.columns) != COLUMNS[1:]:   # COLUMNS[0] is "id", now the index
        problems.append("columns are not the expected 30 measurements plus diagnosis")

    if df.index.duplicated().any():
        problems.append(f"{int(df.index.duplicated().sum())} patient ids appear twice")

    # --- completeness ------------------------------------------------------
    if df.isna().any().any():
        problems.append(f"{int(df.isna().sum().sum())} missing values")

    # --- the labels --------------------------------------------------------
    unexpected = set(df["diagnosis"].unique()) - {"B", "M"}
    if unexpected:
        problems.append(f"diagnosis should only be B or M, found {sorted(unexpected)}")
    else:
        malignant = (df["diagnosis"] == "M").mean()
        low, high = MALIGNANT_FRACTION
        if not low <= malignant <= high:
            problems.append(
                f"{malignant:.1%} malignant, outside the {low:.0%}-{high:.0%} this "
                "model was validated on"
            )

    # --- the biology -------------------------------------------------------
    features = df.drop(columns="diagnosis")

    # Every one of the thirty numbers is a distance, an area, or a ratio.
    # None of them can be negative.
    if (features < 0).any().any():
        problems.append("negative values in measurements that cannot be negative")

    # A nucleus that was measured at all has a size. Zero here is not biology,
    # it is a failed measurement -- unlike zero concavity, below.
    for column in ["radius_mean", "perimeter_mean", "area_mean"]:
        if (df[column] <= 0).any():
            problems.append(f"{column} is zero or less for some samples; no nucleus is")

    # Zero concavity is real, but it is all-or-nothing. A sample that is zero in
    # some concavity columns and not others has been assembled wrongly.
    zeros = (df[SMOOTH_OUTLINE_COLUMNS] == 0)
    part_way = zeros.sum(axis=1).between(1, len(SMOOTH_OUTLINE_COLUMNS) - 1)
    if part_way.any():
        problems.append(
            f"{int(part_way.sum())} samples are zero in some concavity columns but "
            "not all six"
        )

    smooth = zeros.all(axis=1).mean()
    if smooth > MAX_SMOOTH_OUTLINES:
        problems.append(
            f"{smooth:.0%} of samples have no concave points at all; edge detection "
            "may have failed"
        )

    if problems:
        raise ValueError(
            "data contract failed:\n  - " + "\n  - ".join(problems)
        )
    return df
