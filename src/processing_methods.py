"""Stable method selections, independent of EyeFlow's HDF5 root names."""

PROCESSING_ROOTS = {
    "Processing": ("ProcessingMoments", "Processing"),
    "ProcessingAlt": ("ProcessingBandRatio", "ProcessingAlt"),
}
PROCESSING_METHODS = {
    "Processing": "doppler_moments",
    "ProcessingAlt": "frequency_bands",
}
PROCESSING_LABELS = {
    "Processing": "Doppler moments",
    "ProcessingAlt": "Frequency-band ratio",
}
PROCESSING_BRANCHES = (
    "Processing",
    "ProcessingAlt",
    "ProcessingMoments",
    "ProcessingBandRatio",
)


def normalize_processing_branch(value: str) -> str:
    """Keep legacy selection keys while accepting current names and GUI labels."""
    for selection, roots in PROCESSING_ROOTS.items():
        if value in (*roots, PROCESSING_LABELS[selection]):
            return selection
    raise ValueError(
        f"Unknown EyeFlow processing branch {value!r}; "
        f"expected one of {', '.join(PROCESSING_BRANCHES)}."
    )
