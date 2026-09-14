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

# The thirty measurements themselves - every column except the id and the diagnosis.
# This is what the model is handed, and what an incoming batch has to supply.
FEATURES = COLUMNS[2:]

# The ten "mean" columns, used wherever a figure or a table shows one summary only.
MEAN_FEATURES = [f"{measurement}_mean" for measurement in MEASUREMENTS]

DATA_FILE = "data/wdbc.data"

# The column a batch of new samples uses to name each patient. The training
# file calls it the same thing; it is the one column a batch shares with it.
BATCH_INDEX = "id"


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
#
# The rules come in two kinds, and keeping them apart is the point of this
# section. Some are true of ANY sample this model is allowed to score: thirty
# named measurements, nothing missing, nothing negative, no nucleus of zero size.
# Others are true only of THIS cohort: 569 rows, 37% malignant, and a diagnosis
# column to check that against.
#
# A batch of forty new patients breaks the second set by existing. That is not a
# defect in the batch, and a contract that cannot tell the two apart is a
# contract that can only ever be used on the file it was written for.
#
#     batch_problems(df)     what an incoming batch must satisfy to be scored
#     cohort_problems(df)    what the training file must satisfy, and only it
#     validate_batch(df)     raise if a batch breaks the first set
#     validate(df)           raise if the training file breaks either

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

# Below this many samples, that proportion is not evidence of anything. One
# smooth outline in a batch of five is 20%, and it is one patient with a small
# round nucleus. Applying a rate rule to a handful of samples does not make the
# check stricter, it makes it fire on chance -- and a check that cries wolf is a
# check somebody switches off.
MIN_SAMPLES_FOR_A_RATE = 30


def measurements(df):
    """The thirty measurement columns, whether or not a diagnosis travels with them.

    The training file carries a diagnosis. A batch of new patients does not --
    that is the thing being asked for. Both are otherwise the same table.
    """
    return df.drop(columns="diagnosis") if "diagnosis" in df.columns else df


def batch_problems(df):
    """What is wrong with this batch. An empty list means it can be scored.

    These rules do not depend on how many patients arrived, or on how many of
    them turn out to have cancer. They are what the model needs in order to
    return a probability that means anything.
    """
    problems = []
    features = measurements(df)

    # The columns come first, and alone. Every rule below names a column, so
    # running them against a table whose columns are unknown produces a crash
    # instead of an explanation -- and an explanation is the whole product here.
    missing = [column for column in FEATURES if column not in features.columns]
    unexpected = [column for column in features.columns if column not in FEATURES]
    if missing:
        problems.append(f"{len(missing)} measurements missing: {', '.join(missing)}")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected columns: {', '.join(unexpected)}")
    if problems:
        return problems

    if len(df) == 0:
        return ["the batch is empty"]

    if df.index.duplicated().any():
        problems.append(f"{int(df.index.duplicated().sum())} patient ids appear twice")

    # --- completeness ------------------------------------------------------
    if features.isna().any().any():
        problems.append(f"{int(features.isna().sum().sum())} missing values")

    # --- the biology -------------------------------------------------------
    # Every one of the thirty numbers is a distance, an area, or a ratio.
    # None of them can be negative.
    if (features < 0).any().any():
        problems.append("negative values in measurements that cannot be negative")

    # A nucleus that was measured at all has a size. Zero here is not biology,
    # it is a failed measurement -- unlike zero concavity, below.
    for column in ["radius_mean", "perimeter_mean", "area_mean"]:
        if (features[column] <= 0).any():
            problems.append(f"{column} is zero or less for some samples; no nucleus is")

    # Zero concavity is real, but it is all-or-nothing. A sample that is zero in
    # some concavity columns and not others has been assembled wrongly.
    zeros = (features[SMOOTH_OUTLINE_COLUMNS] == 0)
    part_way = zeros.sum(axis=1).between(1, len(SMOOTH_OUTLINE_COLUMNS) - 1)
    if part_way.any():
        problems.append(
            f"{int(part_way.sum())} samples are zero in some concavity columns but "
            "not all six"
        )

    smooth = zeros.all(axis=1).mean()
    if len(df) >= MIN_SAMPLES_FOR_A_RATE and smooth > MAX_SMOOTH_OUTLINES:
        problems.append(
            f"{smooth:.0%} of samples have no concave points at all; edge detection "
            "may have failed"
        )

    return problems


def cohort_problems(df):
    """What is wrong with this table AS THE TRAINING FILE. Empty means it is that file.

    These are the rules nobody can expect a batch of new patients to satisfy.
    They describe one dataset of 569 samples, 37.3% of them malignant, that
    arrived with its answers attached -- and every published number depends on
    the analysis having been run on exactly that.
    """
    problems = []

    if len(df) != N_SAMPLES:
        problems.append(f"expected {N_SAMPLES} rows, found {len(df)}")

    if list(df.columns) != COLUMNS[1:]:   # COLUMNS[0] is "id", now the index
        problems.append("columns are not the expected 30 measurements plus diagnosis")
        return problems

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

    return problems


def _refuse(problems):
    """Raise with every rule that failed, not just the first one.

    A contract that reports one problem at a time makes you fix a bad file in
    ten runs instead of one.
    """
    if problems:
        raise ValueError("data contract failed:\n  - " + "\n  - ".join(problems))


def validate_batch(df):
    """Check a batch of new samples before scoring it. Returns it unchanged.

    This is the contract at the entrance: the model is not shown a single row
    until the batch has said what it is. Nothing here needs a diagnosis, because
    a batch that already had one would not need scoring.
    """
    _refuse(batch_problems(df))
    return df


def validate(df):
    """Check the training file against everything the analysis assumes.

    Returns the table unchanged, so it can be used in place:

        df = validate(load_data())

    Both sets of rules apply here, because this is the one table that is claiming
    to be the dataset the published numbers came from.
    """
    _refuse(batch_problems(df) + cohort_problems(df))
    return df


def read_batch(path):
    """Read a batch of new samples to score. Unlike wdbc.data, it MUST have a header.

    wdbc.data has no header row, so its columns are identified by position. Swap
    two of them and every downstream number is wrong with nothing crashing --
    survivable for one file that never changes and is verified by checksum, and
    not survivable at the entrance to a scoring job, where the file is new every
    time and there is nothing to compare it against.

    So a batch names its columns and they are read BY NAME. The order they arrive
    in stops mattering, which removes the failure rather than guarding against
    it. Anything not recognised is left in place for the contract to report.
    """
    df = pd.read_csv(path)

    if BATCH_INDEX not in df.columns:
        raise ValueError(
            f"a batch must have a '{BATCH_INDEX}' column naming each sample, and a header "
            f"row naming each measurement. Found: {', '.join(map(str, df.columns[:5]))}..."
        )

    df = df.set_index(BATCH_INDEX)
    known = [column for column in FEATURES if column in df.columns]
    rest = [column for column in df.columns if column not in FEATURES]
    return df[known + rest]
