"""Figure conventions for the whole series, set in one place.

    from src.style import use_series_style, BENIGN, MALIGNANT

    use_series_style()

Two colours carry meaning across every figure in every article, so they are
defined once and never re-assigned. The values here are the ones article 2's
published figures were drawn with -- `exploratory_data_analysis.ipynb` decided
them, and this file now agrees with the notebook rather than the other way round.

Only the colours are taken from the notebook. Figure size and resolution stay as
they are below (150 dpi, font size 9), which is the series convention and what
article 3's figures were rendered at.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- the biology ------------------------------------------------------------
# One colour per class, for every figure in every article. A reader who has
# learned that orange means malignant should never have to learn it twice.
BENIGN = "#2a78d6"      # blue
MALIGNANT = "#eb6834"   # orange

CLASS_COLOUR = {"B": BENIGN, "M": MALIGNANT}
CLASS_LABEL = {"B": "Benign", "M": "Malignant"}

# --- the argument -----------------------------------------------------------
# Used in figures about the model rather than the samples: purple is the
# flattering number, grey is the one you can defend.
OPTIMISTIC = "#7a5195"  # training performance
HONEST = "#3f3f3f"      # validated performance

# Everything that is present but not the point -- a null distribution, the bars
# that are not being argued about.
NEUTRAL = "#bdbdbd"

# --- the chrome -------------------------------------------------------------
# Recessive by design: the data should be the darkest thing on the page.
SURFACE = "#fcfcfb"     # the paper
INK = "#0b0b0b"         # labels and titles
MUTED = "#898781"       # axes and tick marks
GRID = "#e1e0d9"        # gridlines

# Diverging map for the correlation matrix: two opposite hues, neutral in the
# middle, so zero correlation reads as nothing rather than as a colour.
CORR_CMAP = LinearSegmentedColormap.from_list("corr", [BENIGN, "#f0efec", "#e34948"])


def use_series_style():
    """Apply the shared figure settings. Call once, near the top of a notebook."""
    plt.rcParams.update({
        # colour, from the notebook
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        # size and resolution, the series convention
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })
