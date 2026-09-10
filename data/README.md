# The raw data

This directory holds an **input**, not an output. Nothing in the analysis writes here.

## What it is

Wisconsin Diagnostic Breast Cancer (WDBC). 569 samples, 30 features, two classes.

The features are computed from a digitised image of a fine-needle aspirate of a breast
mass. Ten measurements are taken of each **cell nucleus** — radius, texture, perimeter,
area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension —
and each is summarised three ways across the nuclei in the image: mean, standard error,
and "worst" (the mean of the three largest values).

**These are measurements of cell nuclei, not of the tumour.** A `radius_mean` of 17.99 is
not an 18 mm lump. Units are not documented in the source.

## Where it came from

| | |
| --- | --- |
| Source | UCI Machine Learning Repository, dataset 17 |
| URL | `https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip` |
| Creators | W.H. Wolberg, W.N. Street, O.L. Mangasarian — University of Wisconsin |
| Donated | November 1995 |
| Retrieved | 2025-08-10 |
| Original publication | Street, Wolberg & Mangasarian, *Nuclear feature extraction for breast tumor diagnosis*, IS&T/SPIE 1993, vol. 1905, 861–870 |

Full source documentation is in `wdbc.names`, which is what makes `wdbc.data` readable
at all.

## Files

| File | | |
| --- | --- | --- |
| `wdbc.data` | 124,103 bytes | 569 rows, 32 comma-separated fields, **no header row** |
| `wdbc.names` | 4,708 bytes | the source documentation |
| `wdbc.zip` | 51,284 bytes | the archive exactly as downloaded from UCI |
| `checksums.sha256` | | the identity of the two files above |

## Identity

```bash
sha256sum -c data/checksums.sha256      # from the repository root
```

Or, doing the same thing and restoring anything missing:

```bash
python -m src.fetch_data --verify
```

`wdbc.data` is `d606af41…`. Every number published in articles 2 and 3 was produced from
that exact file. A file with the right name and different contents is the failure this
guards against, and it is not a hypothetical one — the download-and-extract cells that
used to live in the notebook overwrote these files on every re-run.

## Rules

**The raw files are read-only (`chmod 444`), on purpose.** Do not edit them in place, and
do not "fix" a value here. If the data needs correcting, write a new file with a new name
and a new checksum, and leave the original as it arrived. This is the same reason a
laboratory does not amend the raw instrument output.

**`wdbc.data` has no header row.** Column names are reconstructed in code, and the order
matters — all ten means, then all ten standard errors, then all ten "worst" values:

```python
["id", "diagnosis"] + [f"{m}_{s}" for s in ["mean", "se", "worst"] for m in measurements]
```

Reverse those two loops and every one of the 30 columns is silently mislabelled. Nothing
crashes; the model still trains; the results are wrong.

**Do not switch to UCI's tidied `data.csv`.** It carries a header, renames the columns to
`radius1/2/3`, and moves the diagnosis to the last column. Nothing in this repository
works on it.

**Zero is a real measurement.** Some samples record zero for concavity and concave points.
That is not missing data — a small, smooth nucleus has no detectable concave region. Any
validation rule written against these columns must allow zero.

## Limits

One institution, one era, early-1990s instruments. No patient context and no documented
units. This is a teaching dataset. **Nothing derived from it here is for clinical use.**
