# Model card - WDBC diagnostic classifier

The label that travels with the model. A model card answers the questions somebody asks
when they did not build the thing and have to decide whether to trust it: what does it do,
what was it shown, what does it get wrong, and when should somebody look at it again.

| | |
| --- | --- |
| Model | Standardised 30-feature logistic regression (`StandardScaler` + `LogisticRegression`, one pipeline) |
| Built by | `src/model.py` / `src/run.py` - `make run` |
| Owner | Riccha Sethi |
| Written for | *Research to Production - A Data Science Biologist*, articles 2-4 |
| Card last reviewed | 14 September 2026 |
| **Status** | **Teaching artefact. NOT for clinical use, and not validated for any clinical purpose.** |

There is no version number. The model is rebuilt by every run and `models/model.joblib` is
untracked on purpose, so a fitted model is identified by the SHA-256 recorded in its run
manifest alongside the input checksum, the commit and the environment that produced it.
See `results/` and the run record section of `README.md`.

## What it does

Takes thirty measurements computed from a digitised image of a fine-needle aspirate of a
breast mass and returns **P(malignant)**, a probability between 0 and 1.

The probability becomes one of three outcomes, not two (`decide()` in `src/model.py`):

| P(malignant) | outcome |
| --- | --- |
| below 0.30 | **benign** |
| 0.30 to 0.70 | **refer to a pathologist** - the model is not allowed to answer here |
| above 0.70 | **malignant** |

with the 0.5 threshold separating benign from malignant among the cases it does decide.

The features are measurements of **cell nuclei, not of the tumour**. A `radius_mean` of
17.99 is not an 18 mm lump.

## What it was trained and validated on

| | |
| --- | --- |
| Dataset | Wisconsin Diagnostic Breast Cancer (WDBC), UCI dataset 17 |
| Input file | `data/wdbc.data`, SHA-256 `d606af41...`, read-only, verified before every run |
| Samples | 569 - 357 benign (62.7%), 212 malignant (37.3%) |
| Origin | One institution, University of Wisconsin. Early-1990s instruments. Donated November 1995 |
| Split | 455 train / 114 held out, stratified, `random_state=42` |
| Cross-validation | 5-fold stratified, 10 repeats, on the training half only |
| Referral band | Chosen on cross-validated **training** predictions, never on the test set |

Full provenance, including what the thirty numbers mean and why the raw file is read-only,
is in `data/README.md`.

## Required inputs

Thirty numeric features, named and ordered exactly as `COLUMNS` in `src/load_data.py`
builds them: all ten measurements as `mean`, then all ten as `se`, then all ten as
`worst`, for radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry and fractal dimension - **in that order**. The file has no header
row, so the order is the only thing that identifies a column.

The scaler travels inside the pipeline, so the model expects raw measurements on their
original scale and standardises them itself. Feeding it pre-scaled data is a silent error.

The data contract in `src/load_data.py` states the rest of what the model assumes, in two
parts. `batch_problems()` holds the rules true of any sample it may score: the thirty
measurements correctly named, no duplicate ids, no missing values, nothing negative, no
nucleus of zero size, and the zero-concavity block all-or-nothing in no more than 10% of
samples (a rate applied only to batches of 30 or more, below which it would fire on chance).
`cohort_problems()` holds the rules true only of the training file: 569 rows, a diagnosis
column of `B` and `M`, and 30-45% of them malignant.

**A batch to be scored must carry a header row and an `id` column**, and is read by column
name (`read_batch()`), not by position. `make predict FILE=...` scores nothing until that
batch has passed `batch_problems()`.

## Measured performance

On the 114 held-out patients, at the 0.5 threshold, deciding every case:

| | value | 95% CI |
| --- | --- | --- |
| accuracy | 96.5% | 91.3 - 98.6 |
| recall (cancers found) | 92.9% | 81.0 - 97.5 |
| precision | 97.5% | |
| ROC-AUC | 0.996 | |
| do-nothing baseline (call everyone benign) | 63.2% | |

Cross-validated accuracy on the training half: **97.3% ± 1.5%**.

The four mistakes, counted separately because they are not equally bad:

| | count | what it means for a patient |
| --- | --- | --- |
| true negatives | 71 | correctly sent home |
| false positives | 1 | a benign sample worked up unnecessarily |
| **false negatives** | **3** | **a cancer called benign, and sent home untreated** |
| true positives | 39 | correctly referred on |

**Negative control.** With the labels shuffled, 1,000 times: 59.7% ± 1.5%, best run 64.4%,
p = 0.001. The model is not reading noise.

**With the referral band applied**, once, to the same 114 patients:

| | |
| --- | --- |
| decided automatically | 110 |
| referred to a pathologist | 4 (3.5%) |
| accuracy of the automatic decisions | 99.1% (was 96.5% deciding everything) |
| cancers missed automatically | 1 (was 3) |

The band does not make the model better - every probability is unchanged. It makes the
system better, and it converts errors into work a person has to do: 3.5% of 10,000 slides
a year is 350 extra expert reviews. A band the pathologist cannot absorb is not a safety
feature, it is a queue.

## Known failure modes

**1. A confident mistake the band does not catch.** Patient 859983 is a cancer this model
calls benign at P = 0.061 - 94% confident, and completely wrong. She is nowhere near the
referral band, and no widening of it reaches her without referring half the cohort. Pinned
in `tests/test_referral.py` so nobody forgets it.

> A model can tell you when it is uncertain. It cannot tell you when it is confidently wrong.

**2. Silently mislabelled columns in the training file.** `wdbc.data` has no header, so
column identity is positional. Swapping two columns *within* a block - `radius_mean` and
`texture_mean`, for instance - passes the contract, produces a plausible accuracy, and moves
neither the positive-call rate nor the referral rate enough to notice. Only the input
checksum catches it, which is why that file is verified before every run and kept read-only.

This is **closed for new data**: a batch must carry a header and is read by column name, so
the order it arrives in cannot mislabel anything. The cost of not doing so is measured -
scoring forty patients from the same file positionally instead of by name changes 25 of the
40 diagnoses.

**3. A batch cannot be checked against the population it came from.** The prevalence rule
needs a diagnosis column, and a batch to be scored has none - that is what it is asking for.
So a batch drawn from a screening clinic rather than a referral clinic passes every rule at
the entrance and is scored as though it were the validated population. The positive-call
rate in the batch record is the only signal available, and it is a hint, not a check.

**4. Population change without code change.** Sensitivity and specificity do not carry the
meaning of a positive result across populations; prevalence does. A change in referral
policy alters the population without altering one line of code. This model was validated
on a 37.3% malignant cohort from one hospital and is not validated for screening.

**5. Instrument and scale shift.** The model standardises using training statistics. A new
scanner or recalibrated instrument that shifts the size features degrades accuracy while
still returning a full set of confident-looking probabilities. Nothing in the pipeline
raises an error.

**6. Ground truth is slow.** Accuracy is a late signal. The early signals are the input
distribution, the control patients, the positive-call rate and the referral rate - all of
which are available on the day, without labels.

## Where it should not be used

- **Any clinical setting.** This is a 1990s teaching dataset from a single institution,
  with no patient context, no documented units, and 569 samples.
- **Screening populations**, or any cohort whose malignant fraction sits outside the
  30-45% the contract allows.
- **Data from another institution, scanner or era**, without revalidation.
- **Anywhere the three referred-and-reviewed cases per hundred cannot actually be
  reviewed** by someone qualified.

## Controls, read on every run

Three patients whose predicted probability must not move, to six decimal places
(tolerance 1e-6). They are a fingerprint of the whole pipeline, not a performance
measurement. A run whose controls have drifted writes no manifest.

| patient | expected | why |
| --- | --- | --- |
| 859711 | 0.000000 | negative control - the most certainly benign sample |
| 915143 | 1.000000 | positive control - the most certainly malignant sample |
| 879523 | 0.491610 | borderline control - a cancer called benign, and only just |

The borderline one is the one that matters: a control at 0.0 and a control at 1.0 survive
almost any change, but the sample beside the decision boundary moves as soon as anything
upstream does.

## What triggers a review

Any one of these means stop and look, rather than re-run:

- **a control patient moves** by more than 1e-6;
- **the data contract fails** for any reason;
- **the referral rate departs from 3.5%**, or the positive-call rate from 35% - either
  says the incoming population or the inputs have changed;
- **the malignant fraction leaves 30-45%**;
- **a new instrument, scanner, institution or patient population** starts sending samples;
- **a library version changes** and a headline number moves with it - `requirements.txt`
  is pinned precisely so this is a decision rather than an accident;
- **the published numbers stop reproducing**: 97.3% cross-validated, 96.5% held out,
  3 false negatives and 1 false positive. `make test` checks all four.

A model should not live forever without review. Biological processes, instruments,
populations and software all change, and this one has an expiry date it has already passed
for every purpose except teaching.
