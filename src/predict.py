"""Score a batch of new samples with the model this repository builds.

    python -m src.predict data/incoming/2026-09-14.csv
    make predict FILE=data/incoming/2026-09-14.csv

Run it from the repository root.

This is the other half of production, and the half an analysis repository usually
never grows. `src/run.py` answers "is the model any good?" -- it refits, measures
and reports on data whose answers are already known. Nothing in it can score a
patient nobody has diagnosed yet, which is the only thing anyone actually wants a
model for.

Batch is the right shape for this. Sequencing runs arrive as runs, experimental
batches arrive together, reports go out weekly, and nobody needs a diagnosis in
fifty milliseconds. A scheduled job that looks for new data, checks it, scores it
and records what it did is a perfectly legitimate production system.

Three things make it one rather than a script that prints numbers:

  the contract is at the entrance   nothing is scored until the batch has said
                                    what it is, and it is read BY COLUMN NAME
  the model is the recorded one     loaded from models/model.joblib and named by
                                    checksum, not refitted here on the way past
  the job leaves a record           what was scored, by which model, under which
                                    code and packages, and what came out

It writes no labels it cannot support: every sample lands in benign, malignant,
or "refer to a pathologist", and the referral rate is reported because it is one
of the few things you can watch on the day, months before any label arrives to
say whether today's batch was handled correctly.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from src.fetch_data import sha256
from src.load_data import FEATURES, read_batch, validate_batch
from src.manifest import MODEL_FILE, batch_paths, write_batch_record
from src.model import BAND, THRESHOLD, decide


def rule(title):
    """A heading, so the report can be read at a glance."""
    print(f"\n{title}")
    print("-" * len(title))


def load_model(path=MODEL_FILE):
    """Load the fitted pipeline, and its checksum, or explain why we cannot.

    The model is not rebuilt here. A scoring job that fits its own model is not
    scoring, it is another analysis -- and the thing it scored patients with
    would exist only for as long as the process ran.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. The model is rebuilt and checksummed by the "
            "analysis, so run `make run` first; nothing here refits it."
        )

    return joblib.load(path), sha256(path)


def score(model, batch):
    """Probabilities and three-way decisions for a validated batch.

    The columns are put in the order the model was fitted on, by name. That is
    the whole reason a batch is required to carry a header: a file read by
    position can be scored wrongly without anything going wrong.
    """
    probabilities = model.predict_proba(batch[FEATURES])[:, 1]
    return probabilities, decide(probabilities)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[2].strip())
        print("\nusage: python -m src.predict <batch.csv>")
        return 2

    batch_file = Path(argv[0])
    if not batch_file.exists():
        print(f"STOP. {batch_file} does not exist.")
        return 1

    # 1. The model, before the data. If it is missing, nothing else matters.
    try:
        model, model_sha = load_model()
    except FileNotFoundError as missing:
        print(f"STOP. {missing}")
        return 1

    # 2. The batch, read by column name, and checked before the model sees a row.
    #    A batch that fails here is not scored at all: a plausible probability
    #    computed from a malformed file is the failure this whole repository is
    #    written against, and at the entrance to a scoring job it reaches a
    #    patient rather than a paragraph.
    try:
        batch = validate_batch(read_batch(batch_file))
    except ValueError as broken:
        print(f"STOP. {batch_file} was not scored.\n")
        print(broken)
        print("\nNothing was written. Fix the batch and run this again.")
        return 1

    # 3. The scoring itself, which is the smallest step here on purpose.
    probabilities, decisions = score(model, batch)

    predictions_file, record_file = batch_paths(batch_file)
    pd.DataFrame(
        {"probability": probabilities.round(6), "decision": decisions},
        index=batch.index,
    ).to_csv(predictions_file)

    record = write_batch_record(
        record_file, batch_file, model_sha, decisions, probabilities, predictions_file,
    )

    # 4. The report. The counts first, because they are what somebody has to act
    #    on, and the rates second, because they are what somebody has to watch.
    rule(f"BATCH  {batch_file}")
    print(f"  samples          {len(batch)}")
    print(f"  model            {MODEL_FILE}  sha256 {model_sha[:8]}...")
    print(f"  data contract    passed")

    counts = {outcome: int((decisions == outcome).sum())
              for outcome in ["benign", "refer", "malignant"]}

    rule(f"DECISIONS  (threshold {THRESHOLD}, referral band {BAND[0]} to {BAND[1]})")
    print(f"  benign                 {counts['benign']}")
    print(f"  malignant              {counts['malignant']}")
    print(f"  referred to a human    {counts['refer']}"
          f"   ({counts['refer'] / len(batch):.1%} of this batch)")
    print()
    print(f"  positive-call rate     {counts['malignant'] / len(batch):.1%}")
    print(f"  referral rate          {counts['refer'] / len(batch):.1%}")
    print()
    print("  No accuracy is reported, because nobody knows the answers yet. These")
    print("  two rates are what you can watch today: a large move in either says the")
    print("  incoming population or the measurements have changed, months before a")
    print("  label arrives to confirm it.")

    rule("RECORDED")
    print(f"  {predictions_file}   one row per sample")
    print(f"  {record}   input, model, code and environment, by checksum")

    rule("LIMITS")
    print("  A teaching model, fitted to 569 samples from one hospital in the early")
    print("  1990s. NOTHING HERE IS FOR CLINICAL USE, and a decision printed above")
    print("  is a demonstration of a pipeline, not advice about a patient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
